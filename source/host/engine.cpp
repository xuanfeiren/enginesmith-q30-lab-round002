// Qwen3-30B-A3B inference engine on Trainium2: tensor parallel over 4 logical NeuronCores (the first four visible),
// EngineSmith protocol v1 on stdin / stdout. Every NEFF is embedded (neffs.S). Host: loads and re-lays out the
// checkpoint's weights (values unchanged), schedules NEFFs (one thread per rank), handles the protocol.
//   prefill: emb, 48 x layer (per-layer slices of the decode tensors), head  (buckets 256 / 1,024 / 4,096)
//   decode : one NEFF per step (48 layers + LM head + argmax on the device), attention buckets 512 / 4,608
// Kernels: kernels/step_moe.py (decode), kernels/prefill_tp.py (prefill); layouts: kernels/*_weights.py.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <map>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <immintrin.h>

using std::string;
using std::vector;
typedef uint16_t bf16;

// ---------------------------------------------------------------- embedded NEFFs
extern "C" {
extern const char neff_table_names[];       // "name\0name\0...\0\0"
extern const char* const neff_table_ptr[];   // start, end pairs
}

static std::map<string, std::pair<const char*, size_t>> neffs_embedded() {
  std::map<string, std::pair<const char*, size_t>> m;
  const char* p = neff_table_names;
  int i = 0;
  while (*p) {
    string nm(p);
    m[nm] = {neff_table_ptr[2 * i], (size_t)(neff_table_ptr[2 * i + 1] - neff_table_ptr[2 * i])};
    p += nm.size() + 1;
    i++;
  }
  return m;
}

// ---------------------------------------------------------------- model constants (Qwen3-30B-A3B, TP=4)
static const int W = 4, L = 48, H = 2048, HT = 16, E = 128, NQ = 32, NKV = 4, D = 128, IM = 768, VOCAB = 151936;
static const int C = 2 * W, NQL = NQ / C, NKL = 1, IL = IM / C, ILT = 1, ILP = IL, F1 = HT * 2 * ILT * ILP,
                 F2 = ILT * HT * 128;
static const int TMAX = 4608, VT = (VOCAB + 127) / 128, NVL = (VT + C - 1) / C, NG = 4, LG = 12;
static vector<int> DEC_B;   // decode attention buckets: every embedded dec_b<TB> NEFF, ascending (the last = TMAX)
static const int PF_B[] = {256, 1024, 4096};
static const int PF_MAX = 4096;

static FILE* g_out = nullptr;  // protocol replies (the original stdout)
static void die(const string& s) { fprintf(stderr, "engine: fatal: %s\n", s.c_str()); fflush(stderr); _exit(1); }
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  char b_[512]; snprintf(b_, sizeof b_, "%s:%d %s -> %d (%s)", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); die(b_);} } while (0)
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

// ---------------------------------------------------------------- minimal JSON
struct JV {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false; double n = 0; string s; vector<JV> a; vector<std::pair<string, JV>> o;
  const JV* get(const string& k) const { for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
};
struct JP {
  const char* p; const char* e;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
  [[noreturn]] void fail() { throw std::runtime_error("bad JSON"); }
  string str() {
    if (*p != '"') fail();
    p++; string r;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        p++; if (p >= e) fail();
        char c = *p++;
        switch (c) { case 'n': r += '\n'; break; case 't': r += '\t'; break; case 'r': r += '\r'; break;
          case 'b': r += '\b'; break; case 'f': r += '\f'; break;
          case 'u': { if (e - p < 4) fail(); unsigned v = strtoul(string(p, 4).c_str(), nullptr, 16); p += 4;
                      if (v < 0x80) r += (char)v; else if (v < 0x800) { r += (char)(0xC0 | (v >> 6)); r += (char)(0x80 | (v & 63)); }
                      else { r += (char)(0xE0 | (v >> 12)); r += (char)(0x80 | ((v >> 6) & 63)); r += (char)(0x80 | (v & 63)); } break; }
          default: r += c; }
      } else r += *p++;
    }
    if (p >= e) fail();
    p++; return r;
  }
  JV val() {
    ws(); if (p >= e) fail();
    JV v;
    if (*p == '{') {
      v.t = JV::OBJ; p++; ws();
      if (*p == '}') { p++; return v; }
      while (true) { ws(); string k = str(); ws(); if (*p != ':') fail(); p++; v.o.push_back({k, val()}); ws();
        if (*p == ',') { p++; continue; } if (*p == '}') { p++; break; } fail(); }
    } else if (*p == '[') {
      v.t = JV::ARR; p++; ws();
      if (*p == ']') { p++; return v; }
      while (true) { v.a.push_back(val()); ws(); if (*p == ',') { p++; continue; } if (*p == ']') { p++; break; } fail(); }
    } else if (*p == '"') { v.t = JV::STR; v.s = str(); }
    else if (!strncmp(p, "true", 4)) { v.t = JV::BOOL; v.b = true; p += 4; }
    else if (!strncmp(p, "false", 5)) { v.t = JV::BOOL; p += 5; }
    else if (!strncmp(p, "null", 4)) { p += 4; }
    else { char* q; v.n = strtod(p, &q); if (q == p) fail(); v.t = JV::NUM; p = q; }
    return v;
  }
};
static JV parse_json(const char* s, size_t n) { JP j{s, s + n}; JV v = j.val(); return v; }

// fast path for a long int array: parse "prompt":[...] without building JV objects
static string json_escape(const string& s) {
  string r;
  for (char c : s) { if (c == '"' || c == '\\') { r += '\\'; r += c; } else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", c); r += b; } else r += c; }
  return r;
}

// ---------------------------------------------------------------- safetensors
struct STensor { const uint8_t* data; size_t bytes; vector<int64_t> shape; string dtype; };
struct SafeTensors {
  std::map<string, STensor> t;
  void open_dir(const string& dir) {
    DIR* d = opendir(dir.c_str()); if (!d) die("cannot open model dir " + dir);
    vector<string> files;
    while (auto* en = readdir(d)) { string n = en->d_name; if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors") files.push_back(n); }
    closedir(d);
    if (files.empty()) die("no .safetensors in " + dir);
    for (auto& fn : files) {
      string p = dir + "/" + fn;
      int fd = ::open(p.c_str(), O_RDONLY); if (fd < 0) die("open " + p);
      struct stat st; fstat(fd, &st);
      auto* m = (const uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
      if (m == MAP_FAILED) die("mmap " + p);
      ::close(fd);
      uint64_t n; memcpy(&n, m, 8);
      JV h = parse_json((const char*)m + 8, n);
      for (auto& kv : h.o) {
        if (kv.first == "__metadata__") continue;
        STensor s; s.dtype = kv.second.get("dtype")->s;
        for (auto& x : kv.second.get("shape")->a) s.shape.push_back((int64_t)x.n);
        auto* off = kv.second.get("data_offsets");
        size_t a = (size_t)off->a[0].n, b = (size_t)off->a[1].n;
        s.data = m + 8 + n + a; s.bytes = b - a;
        t[kv.first] = s;
      }
    }
  }
  const bf16* get(const string& name, size_t expect_elems) const {
    auto it = t.find(name); if (it == t.end()) die("missing tensor " + name);
    if (it->second.dtype != "BF16") die(name + " is not BF16");
    if (it->second.bytes != expect_elems * 2) die(name + ": unexpected size");
    return (const bf16*)it->second.data;
  }
};
static inline float bf2f(bf16 v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

// ---------------------------------------------------------------- device state
struct Pool {
  int vnc; std::map<string, nrt_tensor_t*> t; std::map<string, size_t> sz;
  nrt_tensor_t* alloc(const string& name, size_t size) {
    if (t.count(name)) { if (sz[name] != size) die("pool size mismatch " + name); return t[name]; }
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, vnc, size, name.c_str(), &x));
    CK(nrt_tensor_memset(x, 0, 0, size));
    t[name] = x; sz[name] = size; return x;
  }
};
struct Ex { nrt_model_t* m; nrt_tensor_set_t* in; nrt_tensor_set_t* out; };
struct Rank {
  int r; Pool pool;
  std::map<string, nrt_model_t*> models;
  vector<std::array<Ex, 2>> dec;  // [bucket][set A (posX->posY) / B (posY->posX)]
  vector<Ex> pf[3];               // per prefill bucket: emb, L x layer, head
  // job
  vector<Ex> job; std::atomic<int> gen{0}; std::atomic<int> done{0}; string err;
  std::thread th;
};
static vector<Rank*> R;
static std::atomic<bool> g_quit{false};

static void rank_loop(Rank* k) {
  int seen = 0;
  auto idle_since = clk::now();
  while (!g_quit.load()) {
    int g = k->gen.load(std::memory_order_acquire);
    if (g == seen) {
      if (ms_since(idle_since) > 10000) usleep(200); else _mm_pause();
      continue;
    }
    seen = g;
    k->err.clear();
    vector<NRT_STATUS> rets(k->job.size(), NRT_SUCCESS);
    nrta_seq_t seq = 0; NRT_STATUS s;
    bool ok = true;
    for (size_t j = 0; j < k->job.size(); j++) {
      while ((s = nrta_execute_schedule(k->job[j].m, k->job[j].in, k->job[j].out, 0, &rets[j], &seq)) == NRT_QUEUE_FULL) _mm_pause();
      if (s != NRT_SUCCESS) { k->err = string("schedule failed: ") + nrt_get_status_as_str(s); ok = false; break; }
    }
    if (ok && !k->job.empty()) {
      bool fin = false;
      while (!fin) { s = nrta_is_completed(seq, &fin); if (s != NRT_SUCCESS) { k->err = "is_completed failed"; break; } }
      for (auto x : rets) if (x != NRT_SUCCESS) { k->err = string("execution failed: ") + nrt_get_status_as_str(x); break; }
    }
    idle_since = clk::now();
    k->done.store(g, std::memory_order_release);
  }
}

static string run_jobs(const vector<vector<Ex>>& jobs) {
  for (int r = 0; r < W; r++) R[r]->job = jobs[r];
  for (int r = 0; r < W; r++) R[r]->gen.fetch_add(1, std::memory_order_release);
  for (int r = 0; r < W; r++) { int g = R[r]->gen.load(); while (R[r]->done.load(std::memory_order_acquire) != g) _mm_pause(); }
  for (int r = 0; r < W; r++) if (!R[r]->err.empty()) return "rank " + std::to_string(r) + ": " + R[r]->err;
  return "";
}

// ---------------------------------------------------------------- weight preparation (values unchanged)
static void parallel_for(int n, int threads, const std::function<void(int)>& f) {
  std::atomic<int> next{0};
  vector<std::thread> th;
  for (int t = 0; t < threads; t++) th.emplace_back([&] { int i; while ((i = next.fetch_add(1)) < n) f(i); });
  for (auto& x : th) x.join();
}

static string lname(int l, const string& n) { return "model.layers." + std::to_string(l) + "." + n; }

// ws tile: dst[p * ld + kt*128 + j] = W[(row0 + j) * K + kt*128 + p], for kt < KT (one 128-row tile of W [N, K])
static void ws_tile(bf16* dst, size_t ld, const bf16* Wt, int K, int row0) {
  int KT = K / 128;
  for (int j = 0; j < 128; j++) {
    const bf16* src = Wt + (size_t)(row0 + j) * K;
    for (int kt = 0; kt < KT; kt++)
      for (int p = 0; p < 128; p++) dst[(size_t)p * ld + kt * 128 + j] = src[kt * 128 + p];
  }
}

static void load_weights(const SafeTensors& st) {
  auto t0 = clk::now();
  // ---- allocate (decode tensors; the prefill NEFFs bind slices of these)
  size_t sz_wqkv_l = (size_t)2 * 128 * (NQL + 2 * NKL) * HT * 128 * 2, sz_wo_l = (size_t)2 * 128 * HT * NQL * 128 * 2;
  size_t sz_wgu_l = (size_t)E * 2 * 128 * F1 * 2, sz_wd_l = (size_t)E * 2 * ILP * F2 * 2;
  for (int r = 0; r < W; r++) {
    Pool& P = R[r]->pool;
    P.alloc("posX", 4); P.alloc("posY", 4); P.alloc("plen", 4); P.alloc("plen_pf", 4);
    P.alloc("prompt", TMAX * 4); P.alloc("toks", (TMAX + 1) * 4);
    P.alloc("embed", (size_t)VOCAB * H * 2);
    P.alloc("ln1", (size_t)L * 128 * HT * 4); P.alloc("ln2", (size_t)L * 128 * HT * 4);
    P.alloc("qn", (size_t)L * 128 * 4); P.alloc("kn", (size_t)L * 128 * 4);
    P.alloc("wqkv", L * sz_wqkv_l); P.alloc("wo", L * sz_wo_l);
    P.alloc("wr", (size_t)L * 128 * HT * E * 2);
    for (int g = 0; g < NG; g++) { P.alloc("wgu" + std::to_string(g), LG * sz_wgu_l); P.alloc("wd" + std::to_string(g), LG * sz_wd_l); }
    P.alloc("normf", 128 * HT * 4);
    P.alloc("wlm", (size_t)2 * 128 * NVL * HT * 128 * 2);
    P.alloc("rot", 128 * 128 * 2);
    P.alloc("cos", (size_t)TMAX * 64 * 4); P.alloc("sin", (size_t)TMAX * 64 * 4);
    P.alloc("cosT", (size_t)128 * TMAX * 4); P.alloc("sinT", (size_t)128 * TMAX * 4);
    P.alloc("kcache", (size_t)L * 2 * NKL * 128 * TMAX * 2); P.alloc("vcache", (size_t)L * 2 * TMAX * NKL * 128 * 2);
    P.alloc("rkb", 128 * 4);
  }
  fprintf(stderr, "engine: allocated in %.0f ms\n", ms_since(t0));
  auto wr_all = [&](const string& name, const void* buf, size_t off, size_t size) {
    for (int r = 0; r < W; r++) CK(nrt_tensor_write(R[r]->pool.t[name], buf, off, size));
  };
  // ---- small / replicated tensors
  {
    const bf16* emb = st.get("model.embed_tokens.weight", (size_t)VOCAB * H);
    parallel_for(W, W, [&](int r) { CK(nrt_tensor_write(R[r]->pool.t["embed"], emb, 0, (size_t)VOCAB * H * 2)); });
    vector<float> ln1((size_t)L * 128 * HT), ln2((size_t)L * 128 * HT), qn((size_t)L * 128), kn((size_t)L * 128);
    for (int l = 0; l < L; l++) {
      const bf16* a = st.get(lname(l, "input_layernorm.weight"), H);
      const bf16* b = st.get(lname(l, "post_attention_layernorm.weight"), H);
      for (int p = 0; p < 128; p++) for (int kt = 0; kt < HT; kt++) {
        ln1[((size_t)l * 128 + p) * HT + kt] = bf2f(a[kt * 128 + p]);
        ln2[((size_t)l * 128 + p) * HT + kt] = bf2f(b[kt * 128 + p]);
      }
      const bf16* q = st.get(lname(l, "self_attn.q_norm.weight"), D);
      const bf16* k = st.get(lname(l, "self_attn.k_norm.weight"), D);
      for (int p = 0; p < 128; p++) { qn[l * 128 + p] = bf2f(q[p]); kn[l * 128 + p] = bf2f(k[p]); }
    }
    wr_all("ln1", ln1.data(), 0, ln1.size() * 4); wr_all("ln2", ln2.data(), 0, ln2.size() * 4);
    wr_all("qn", qn.data(), 0, qn.size() * 4); wr_all("kn", kn.data(), 0, kn.size() * 4);
    const bf16* nf = st.get("model.norm.weight", H);
    vector<float> normf(128 * HT);
    for (int p = 0; p < 128; p++) for (int kt = 0; kt < HT; kt++) normf[p * HT + kt] = bf2f(nf[kt * 128 + p]);
    wr_all("normf", normf.data(), 0, normf.size() * 4);
    vector<bf16> rot(128 * 128, 0);
    for (int m = 0; m < 64; m++) { rot[(m + 64) * 128 + m] = 0xBF80; rot[m * 128 + m + 64] = 0x3F80; }
    wr_all("rot", rot.data(), 0, rot.size() * 2);
    vector<float> cs((size_t)TMAX * 64), sn((size_t)TMAX * 64), csT((size_t)128 * TMAX), snT((size_t)128 * TMAX);
    for (int i = 0; i < 64; i++) {
      double inv = 1.0 / std::pow(1000000.0, (double)(2 * i) / 128.0);
      for (int t = 0; t < TMAX; t++) {
        double f = (double)t * inv;
        cs[(size_t)t * 64 + i] = (float)std::cos(f); sn[(size_t)t * 64 + i] = (float)std::sin(f);
      }
    }
    for (int i = 0; i < 128; i++) for (int t = 0; t < TMAX; t++) { csT[(size_t)i * TMAX + t] = cs[(size_t)t * 64 + (i % 64)]; snT[(size_t)i * TMAX + t] = sn[(size_t)t * 64 + (i % 64)]; }
    wr_all("cos", cs.data(), 0, cs.size() * 4); wr_all("sin", sn.data(), 0, sn.size() * 4);
    wr_all("cosT", csT.data(), 0, csT.size() * 4); wr_all("sinT", snT.data(), 0, snT.size() * 4);
    for (int r = 0; r < W; r++) {
      vector<float> rkb(128, (float)(r * 2 * NVL * 128));
      CK(nrt_tensor_write(R[r]->pool.t["rkb"], rkb.data(), 0, 512));
    }
  }
  fprintf(stderr, "engine: small tensors in %.0f ms\n", ms_since(t0));
  // ---- router, attention, LM head
  parallel_for(L, 16, [&](int l) {
    const bf16* Wr = st.get(lname(l, "mlp.gate.weight"), (size_t)E * H);
    vector<bf16> wr((size_t)128 * HT * E);
    for (int e = 0; e < E; e++) for (int kt = 0; kt < HT; kt++) for (int p = 0; p < 128; p++)
      wr[((size_t)p * HT + kt) * E + e] = Wr[(size_t)e * H + kt * 128 + p];
    wr_all("wr", wr.data(), (size_t)l * wr.size() * 2, wr.size() * 2);
    const bf16* Wq = st.get(lname(l, "self_attn.q_proj.weight"), (size_t)NQ * D * H);
    const bf16* Wk = st.get(lname(l, "self_attn.k_proj.weight"), (size_t)NKV * D * H);
    const bf16* Wv = st.get(lname(l, "self_attn.v_proj.weight"), (size_t)NKV * D * H);
    const bf16* Wo = st.get(lname(l, "self_attn.o_proj.weight"), (size_t)H * NQ * D);
    const int NQKV = NQL + 2 * NKL;
    size_t ldq = (size_t)NQKV * HT * 128, ldo = (size_t)HT * NQL * 128;
    vector<bf16> qkv(2 * 128 * ldq), wo(2 * 128 * ldo);
    for (int r = 0; r < W; r++) {
      for (int c = 0; c < 2; c++) {
        int v = 2 * r + c, q0 = v * NQL, k0 = (v * NKV) / C;
        bf16* dq = qkv.data() + (size_t)c * 128 * ldq;
        for (int i = 0; i < NQL; i++) ws_tile(dq + (size_t)i * HT * 128, ldq, Wq, H, (q0 + i) * 128);
        ws_tile(dq + (size_t)NQL * HT * 128, ldq, Wk, H, k0 * 128);
        ws_tile(dq + (size_t)(NQL + 1) * HT * 128, ldq, Wv, H, k0 * 128);
        // wo[c][p][(nt*NQL + qi)*128 + j] = Wo[nt*128 + j][(q0+qi)*128 + p]
        bf16* dO = wo.data() + (size_t)c * 128 * ldo;
        for (int nt = 0; nt < HT; nt++) for (int j = 0; j < 128; j++) {
          const bf16* src = Wo + (size_t)(nt * 128 + j) * (NQ * D);
          for (int qi = 0; qi < NQL; qi++) for (int p = 0; p < 128; p++)
            dO[(size_t)p * ldo + (nt * NQL + qi) * 128 + j] = src[(q0 + qi) * 128 + p];
        }
      }
      CK(nrt_tensor_write(R[r]->pool.t["wqkv"], qkv.data(), (size_t)l * qkv.size() * 2, qkv.size() * 2));
      CK(nrt_tensor_write(R[r]->pool.t["wo"], wo.data(), (size_t)l * wo.size() * 2, wo.size() * 2));
    }
  });
  fprintf(stderr, "engine: router + attention in %.0f ms\n", ms_since(t0));
  {
    const bf16* LM = st.get("lm_head.weight", (size_t)VOCAB * H);
    size_t ld = (size_t)NVL * HT * 128;
    parallel_for(W * 2, 8, [&](int i) {
      int r = i / 2, c = i % 2;
      vector<bf16> w((size_t)128 * ld, 0);
      for (int t = 0; t < NVL; t++) {
        int vt = NVL * (2 * r + c) + t;
        if (vt < VT) ws_tile(w.data() + (size_t)t * HT * 128, ld, LM, H, vt * 128);
      }
      CK(nrt_tensor_write(R[r]->pool.t["wlm"], w.data(), (size_t)c * w.size() * 2, w.size() * 2));
    });
  }
  fprintf(stderr, "engine: LM head in %.0f ms\n", ms_since(t0));
  // ---- experts: per (layer, rank) job
  parallel_for(L * W, 48, [&](int job) {
    int l = job / W, r = job % W;
    vector<bf16> gu((size_t)E * 2 * 128 * F1), dn((size_t)E * 2 * ILP * F2);
    for (int e = 0; e < E; e++) {
      string pre = "mlp.experts." + std::to_string(e) + ".";
      const bf16* G = st.get(lname(l, pre + "gate_proj.weight"), (size_t)IM * H);
      const bf16* U = st.get(lname(l, pre + "up_proj.weight"), (size_t)IM * H);
      const bf16* Dn = st.get(lname(l, pre + "down_proj.weight"), (size_t)H * IM);
      for (int c = 0; c < 2; c++) {
        int row0 = (2 * r + c) * IL;
        // gu[e][c][p][kt][g][j] = (g ? U : G)[row0 + j][kt*128 + p]   (ILT = 1)
        bf16* d = gu.data() + ((size_t)e * 2 + c) * 128 * F1;
        for (int g = 0; g < 2; g++) {
          const bf16* S = g ? U : G;
          for (int j = 0; j < ILP; j++) {
            const bf16* src = S + (size_t)(row0 + j) * H;
            for (int kt = 0; kt < HT; kt++) for (int p = 0; p < 128; p++)
              d[(size_t)p * F1 + kt * (2 * ILP) + g * ILP + j] = src[kt * 128 + p];
          }
        }
        // dn[e][c][j][nt*128 + q] = Dn[nt*128 + q][row0 + j]
        bf16* dd = dn.data() + ((size_t)e * 2 + c) * ILP * F2;
        for (int n = 0; n < H; n++) {
          const bf16* src = Dn + (size_t)n * IM + row0;
          for (int j = 0; j < ILP; j++) dd[(size_t)j * F2 + n] = src[j];
        }
      }
    }
    string g = std::to_string(l / LG);
    CK(nrt_tensor_write(R[r]->pool.t["wgu" + g], gu.data(), (size_t)(l % LG) * gu.size() * 2, gu.size() * 2));
    CK(nrt_tensor_write(R[r]->pool.t["wd" + g], dn.data(), (size_t)(l % LG) * dn.size() * 2, dn.size() * 2));
  });
  fprintf(stderr, "engine: experts in %.0f ms\n", ms_since(t0));
}

// ---------------------------------------------------------------- NEFF binding
static const char* PER_LAYER[] = {"ln1", "qn", "kn", "wqkv", "wo", "ln2", "wr", "kcache", "vcache"};
static bool is_per_layer(const string& n) { for (auto* p : PER_LAYER) if (n == p) return true; return false; }
static bool is_pf_scratch(const string& n) {
  return n == "ids" || n == "s_h" || n == "s_q" || n == "s_k" || n == "s_v" || n == "s_o" || n == "s_p" || n == "x_tm" ||
         n == "stag" || n == "y_slot" || n == "y_part";
}

static nrt_tensor_t* slice(Pool& P, const string& pname, size_t off, size_t size, const char* nm) {
  auto it = P.t.find(pname); if (it == P.t.end()) die("no pool tensor " + pname);
  if (off + size > P.sz[pname]) die("slice out of range " + pname);
  nrt_tensor_t* x; CK(nrt_tensor_allocate_slice(it->second, off, size, nm, &x)); return x;
}

// prefill NEFF of bucket S (layer >= 0: the layer NEFF bound to layer `layer`; -1: emb / head)
static Ex bind_pf(Rank* k, nrt_model_t* m, int S, int layer) {
  Ex e{m, nullptr, nullptr};
  CK(nrt_allocate_tensor_set(&e.in)); CK(nrt_allocate_tensor_set(&e.out));
  nrt_tensor_info_array_t* inf; CK(nrt_get_model_tensor_info(m, &inf));
  Pool& P = k->pool;
  int lay = layer < 0 ? 0 : layer;
  for (uint64_t i = 0; i < inf->tensor_count; i++) {
    auto& ti = inf->tensor_array[i]; bool in = ti.usage == NRT_TENSOR_USAGE_INPUT; string nm = ti.name;
    nrt_tensor_t* x;
    if (nm == "output_0") x = slice(P, "posX", 0, ti.size, ti.name);
    else if (nm == "plen") x = slice(P, "plen_pf", 0, ti.size, ti.name);
    else if (is_pf_scratch(nm)) { x = P.alloc(nm + "@" + std::to_string(S), ti.size); }
    else if (is_per_layer(nm)) {
      if (P.sz[nm] != (size_t)L * ti.size) die("per-layer size mismatch " + nm);
      x = slice(P, nm, (size_t)lay * ti.size, ti.size, ti.name);
    } else if (nm == "wgu" || nm == "wd") {
      string g = nm + std::to_string(lay / LG);
      if (P.sz[g] != (size_t)LG * ti.size) die("group size mismatch " + nm);
      x = slice(P, g, (size_t)(lay % LG) * ti.size, ti.size, ti.name);
    } else {
      if (P.t.count(nm) && P.sz[nm] == ti.size) x = P.t[nm];
      else { fprintf(stderr, "engine: note: prefill tensor %s (%zu B) allocated on its own\n", nm.c_str(), (size_t)ti.size);
             x = P.alloc(nm + "@" + std::to_string(S), ti.size); }
    }
    CK(nrt_add_tensor_to_tensor_set(in ? e.in : e.out, ti.name, x));
  }
  return e;
}

static void bind_dec(Rank* k, nrt_model_t* m, Ex* sets) {
  Pool& P = k->pool;
  nrt_tensor_info_array_t* inf; CK(nrt_get_model_tensor_info(m, &inf));
  for (int s = 0; s < 2; s++) {
    sets[s].m = m; CK(nrt_allocate_tensor_set(&sets[s].in)); CK(nrt_allocate_tensor_set(&sets[s].out));
    for (uint64_t i = 0; i < inf->tensor_count; i++) {
      auto& ti = inf->tensor_array[i]; bool in = ti.usage == NRT_TENSOR_USAGE_INPUT; string nm = ti.name;
      nrt_tensor_t* x;
      if (nm == "pos") x = P.t[s == 0 ? "posX" : "posY"];
      else if (nm == "output_0") x = P.t[s == 0 ? "posY" : "posX"];
      else {
        if (P.t.count(nm) && P.sz[nm] == ti.size) x = P.t[nm];
        else { fprintf(stderr, "engine: note: decode tensor %s (%zu B) allocated on its own\n", nm.c_str(), (size_t)ti.size);
               x = P.alloc(nm + "@dec", ti.size); }
      }
      CK(nrt_add_tensor_to_tensor_set(in ? sets[s].in : sets[s].out, ti.name, x));
    }
  }
}

// ---------------------------------------------------------------- generation
struct State { bool valid = false; int pos = 0; bool curX = true; };
static State g_st;

static int pf_bucket(int n) { for (int i = 0; i < 3; i++) if (n <= PF_B[i]) return i; return 2; }
static int dec_bucket(int pos) {
  for (size_t i = 0; i + 1 < DEC_B.size(); i++) if (pos < DEC_B[i]) return (int)i;
  return (int)DEC_B.size() - 1;
}

static void write_all_i32(const string& name, const int32_t* v, size_t n, size_t off_elems = 0) {
  for (int r = 0; r < W; r++) CK(nrt_tensor_write(R[r]->pool.t[name], v, off_elems * 4, n * 4));
}

// append n decode steps starting at position pos to the jobs
static void add_decode(vector<vector<Ex>>& jobs, int pos, int n) {
  for (int i = 0; i < n; i++) {
    int b = dec_bucket(pos + i);
    int s = g_st.curX ? 0 : 1;
    for (int r = 0; r < W; r++) jobs[r].push_back(R[r]->dec[b][s]);
    g_st.curX = !g_st.curX;
  }
}

static vector<int> read_toks(int from, int n) {
  vector<int32_t> v(n);
  CK(nrt_tensor_read(R[0]->pool.t["toks"], v.data(), (size_t)from * 4, (size_t)n * 4));
  return vector<int>(v.begin(), v.end());
}

static vector<int> generate(const vector<int>& prompt, int n, bool reset, string& err) {
  vector<int> out;
  if (n < 1) { err = "max_tokens must be at least 1"; return out; }
  vector<vector<Ex>> jobs(W);
  if (reset) {
    int P = (int)prompt.size();
    if (P < 1) { err = "a reset request needs a prompt"; return out; }
    if (P + n > TMAX) { err = "prompt + max_tokens exceed the compiled positions (" + std::to_string(TMAX) + ")"; return out; }
    for (int t : prompt) if (t < 0 || t >= VOCAB) { err = "token id out of range"; return out; }
    g_st.valid = false;
    // the reference (NxD Inference) pads a one-token prompt with token 0 and predicts at the pad position
    vector<int32_t> pf(prompt.begin(), prompt.end());
    if (P == 1) pf.push_back(0);
    int npf = std::min((int)pf.size(), PF_MAX);
    int bi = pf_bucket(npf), S = PF_B[bi];
    vector<int32_t> ids(S, 0);
    for (int i = 0; i < npf; i++) ids[i] = pf[i];
    vector<int32_t> pr(prompt.begin(), prompt.end());
    int32_t plen = P, plen_pf = npf;
    for (int r = 0; r < W; r++) {
      Pool& Pp = R[r]->pool;
      CK(nrt_tensor_write(Pp.t["ids@" + std::to_string(S)], ids.data(), 0, (size_t)S * 4));
      CK(nrt_tensor_write(Pp.t["prompt"], pr.data(), 0, pr.size() * 4));
      CK(nrt_tensor_write(Pp.t["plen"], &plen, 0, 4));
      CK(nrt_tensor_write(Pp.t["plen_pf"], &plen_pf, 0, 4));
    }
    for (int r = 0; r < W; r++) for (auto& e : R[r]->pf[bi]) jobs[r].push_back(e);
    g_st.curX = true;
    if (P == 1) {
      // prediction after [t, 0] lands in toks[2]; the state is: position 1 next, its token = that prediction
      err = run_jobs(jobs); if (!err.empty()) return out;
      jobs.assign(W, {});
      vector<int> t2 = read_toks(2, 1);
      int32_t tok = t2[0], one = 1;
      write_all_i32("toks", &tok, 1, 1);
      for (int r = 0; r < W; r++) CK(nrt_tensor_write(R[r]->pool.t["posX"], &one, 0, 4));
    } else if (P > npf) {
      add_decode(jobs, npf, P - npf);   // the rest of the prompt through the decode step (prompt[pos] while pos < plen)
    }
    add_decode(jobs, P, n - 1);
    err = run_jobs(jobs); if (!err.empty()) return out;
    out = read_toks(P, n);
    g_st.valid = true; g_st.pos = P + n - 1;
  } else {
    if (!prompt.empty()) { err = "a continuation has an empty prompt"; return out; }
    if (!g_st.valid) { err = "a continuation must follow a generate request"; return out; }
    if (g_st.pos + n > TMAX) { err = "continuation exceeds the compiled positions (" + std::to_string(TMAX) + ")"; return out; }
    add_decode(jobs, g_st.pos, n);
    err = run_jobs(jobs); if (!err.empty()) { g_st.valid = false; return out; }
    out = read_toks(g_st.pos + 1, n);
    g_st.pos += n;
  }
  for (int t : out) if (t < 0 || t >= VOCAB) { err = "device produced an out-of-range token"; g_st.valid = false; return {}; }
  return out;
}

// ---------------------------------------------------------------- startup
static void startup(const string& model_dir) {
  auto t0 = clk::now();
  // use (and claim) only the first W visible cores: the runtime claims every core it can see, and the cores after
  // ours belong to the reference's judge
  {
    const char* vis = getenv("NEURON_RT_VISIBLE_CORES");
    vector<int> cores;
    string v = vis ? vis : "0";
    size_t a = 0;
    while (a <= v.size()) {
      size_t b = v.find(',', a); if (b == string::npos) b = v.size();
      string part = v.substr(a, b - a);
      size_t dash = part.find('-');
      if (!part.empty()) {
        int lo = atoi(part.substr(0, dash).c_str()), hi = dash == string::npos ? lo : atoi(part.substr(dash + 1).c_str());
        for (int c = lo; c <= hi; c++) cores.push_back(c);
      }
      a = b + 1;
    }
    if ((int)cores.size() < W) die("need " + std::to_string(W) + " visible NeuronCores, NEURON_RT_VISIBLE_CORES=" + v);
    for (int i = 1; i < W; i++) if (cores[i] != cores[0] + i) die("the first visible NeuronCores are not contiguous: " + v);
    string mine = std::to_string(cores[0]) + "-" + std::to_string(cores[0] + W - 1);
    setenv("NEURON_RT_VISIBLE_CORES", mine.c_str(), 1);
    fprintf(stderr, "engine: NEURON_RT_VISIBLE_CORES=%s -> %s\n", v.c_str(), mine.c_str());
  }
  if (!getenv("NEURON_RT_ROOT_COMM_ID")) {
    // one port per process (collectives bootstrap on localhost)
    int port = 40000 + (getpid() % 20000);
    char b[64]; snprintf(b, sizeof b, "localhost:%d", port); setenv("NEURON_RT_ROOT_COMM_ID", b, 1);
  }
  setenv("NEURON_RT_DISABLE_EXECUTION_BARRIER", "1", 0);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  SafeTensors st; st.open_dir(model_dir);
  auto neffs = neffs_embedded();
  for (int r = 0; r < W; r++) { R.push_back(new Rank()); R[r]->r = r; R[r]->pool.vnc = r; }
  vector<string> names;
  for (auto& kv : neffs) if (kv.first.rfind("dec_b", 0) == 0) DEC_B.push_back(atoi(kv.first.c_str() + 5));
  std::sort(DEC_B.begin(), DEC_B.end());
  if (DEC_B.empty() || DEC_B.back() != TMAX) die("the last decode bucket must be TMAX");
  // load order: the prefill NEFFs first (measured: decode NEFFs loaded first made the 4,096 prefill 1.8% slower,
  // device memory placement); ENGINE_PF_REV (experiments): prefill buckets largest first
  {
    vector<int> pb(PF_B, PF_B + 3);
    if (getenv("ENGINE_PF_REV")) std::reverse(pb.begin(), pb.end());
    for (int S : pb) for (const char* s : {"emb", "layer", "head"}) names.push_back("pf" + std::to_string(S) + "_" + s);
    for (int b : DEC_B) names.push_back("dec_b" + std::to_string(b));
  }
  for (auto& nm : names) {
    if (!neffs.count(nm)) die("NEFF not embedded: " + nm);
    auto& b = neffs[nm];
    for (int r = 0; r < W; r++) CK(nrt_load_collectives(b.first, b.second, r, -1, r, W, &R[r]->models[nm]));
  }
  fprintf(stderr, "engine: %zu NEFFs loaded on %d ranks in %.0f ms\n", names.size(), W, ms_since(t0));
  load_weights(st);
  for (int r = 0; r < W; r++) {
    Rank* k = R[r];
    k->dec.resize(DEC_B.size());
    for (size_t b = 0; b < DEC_B.size(); b++) bind_dec(k, k->models["dec_b" + std::to_string(DEC_B[b])], k->dec[b].data());
    for (int i = 0; i < 3; i++) {
      string s = "pf" + std::to_string(PF_B[i]);
      k->pf[i].push_back(bind_pf(k, k->models[s + "_emb"], PF_B[i], -1));
      for (int l = 0; l < L; l++) k->pf[i].push_back(bind_pf(k, k->models[s + "_layer"], PF_B[i], l));
      k->pf[i].push_back(bind_pf(k, k->models[s + "_head"], PF_B[i], -1));
    }
  }
  for (int r = 0; r < W; r++) R[r]->th = std::thread(rank_loop, R[r]);
  fprintf(stderr, "engine: ready in %.0f ms\n", ms_since(t0));
  // warm up: every prefill bucket and both decode buckets once (the first executions after a load are slow)
  string err;
  for (int S : PF_B) {
    vector<int> p(std::min(S, 600), 0);
    for (size_t i = 0; i < p.size(); i++) p[i] = (int)(i * 7919 % 150000);
    if (S == 4096) p.resize(4000, 11);
    generate(p, 2, true, err);
    if (!err.empty()) die("warm-up failed: " + err);
  }
  for (int b : DEC_B) {   // every decode bucket: a few steps just below its end
    vector<int> p(b - 12, 13);
    generate(p, 8, true, err);
    if (!err.empty()) die("warm-up failed: " + err);
  }
  g_st.valid = false;
  fprintf(stderr, "engine: warm-up done, %.0f ms since start\n", ms_since(t0));
}

// ---------------------------------------------------------------- protocol
static bool read_line(string& line) {
  static string buf;
  static size_t start = 0;
  auto last_input = clk::now();
  while (true) {
    size_t nl = buf.find('\n', start);
    if (nl != string::npos) { line = buf.substr(start, nl - start); start = nl + 1; if (start > (1 << 20)) { buf.erase(0, start); start = 0; } return true; }
    // spin on a non-blocking stdin for a while after the last input, then block
    struct pollfd pf{0, POLLIN, 0};
    int to = ms_since(last_input) < 10000 ? 0 : -1;
    int rc = poll(&pf, 1, to);
    if (rc == 0) { _mm_pause(); continue; }
    if (rc < 0) { if (errno == EINTR) continue; return false; }
    char tmp[1 << 16];
    ssize_t n = read(0, tmp, sizeof tmp);
    if (n == 0) { if (start < buf.size()) { line = buf.substr(start); buf.clear(); start = 0; return true; } return false; }
    if (n < 0) { if (errno == EINTR || errno == EAGAIN) continue; return false; }
    buf.append(tmp, n);
    last_input = clk::now();
  }
}

static void reply(const string& s) { fputs(s.c_str(), g_out); fputc('\n', g_out); fflush(g_out); }

int main(int argc, char** argv) {
  string model;
  for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
  if (model.empty()) { fprintf(stderr, "usage: engine --model MODEL_DIR\n"); return 2; }
  // protocol replies go to a private copy of stdout; fd 1 (anything libnrt prints) goes to stderr
  int outfd = dup(1); dup2(2, 1);
  g_out = fdopen(outfd, "w");
  bool loaded = false;
  string line;
  while (read_line(line)) {
    if (line.find_first_not_of(" \t\r") == string::npos) continue;
    string idtxt = "null";
    try {
      JV req = parse_json(line.data(), line.size());
      const JV* id = req.get("id");
      if (id && id->t == JV::NUM) { char b[32]; snprintf(b, 32, "%lld", (long long)id->n); idtxt = b; }
      const JV* op = req.get("op");
      string o = op && op->t == JV::STR ? op->s : "";
      if (o == "hello") {
        if (!loaded) { startup(model); loaded = true; }
        reply("{\"id\":" + idtxt + ",\"ok\":true,\"protocol\":1}");
      } else if (o == "generate") {
        if (!loaded) { startup(model); loaded = true; }
        vector<int> prompt;
        if (const JV* p = req.get("prompt")) for (auto& x : p->a) prompt.push_back((int)x.n);
        const JV* mt = req.get("max_tokens"); const JV* rs = req.get("reset");
        int n = mt ? (int)mt->n : 0; bool reset = rs ? rs->b : true;
        auto t0 = clk::now();
        string err;
        vector<int> toks = generate(prompt, n, reset, err);
        if (!err.empty()) { reply("{\"id\":" + idtxt + ",\"ok\":false,\"error\":\"" + json_escape(err) + "\"}"); continue; }
        string s = "{\"id\":" + idtxt + ",\"ok\":true,\"tokens\":[";
        for (size_t i = 0; i < toks.size(); i++) { if (i) s += ','; s += std::to_string(toks[i]); }
        s += "]}";
        reply(s);
        if (getenv("ENGINE_VERBOSE")) fprintf(stderr, "engine: generate P=%zu n=%d reset=%d %.2f ms\n", prompt.size(), n, (int)reset, ms_since(t0));
      } else {
        reply("{\"id\":" + idtxt + ",\"ok\":false,\"error\":\"unknown op\"}");
      }
    } catch (std::exception& e) {
      reply("{\"id\":" + idtxt + ",\"ok\":false,\"error\":\"" + json_escape(e.what()) + "\"}");
    }
  }
  g_quit = true;
  for (auto* k : R) if (k->th.joinable()) k->th.join();
  return 0;
}
