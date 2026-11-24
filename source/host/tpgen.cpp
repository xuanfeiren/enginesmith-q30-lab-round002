// tpgen: tensor-parallel layered prefill + decode chain on W logical NeuronCores, one process, one thread per rank.
//   NEURON_RT_ROOT_COMM_ID=localhost:<port> tpgen <emb.neff,layer.neff,head.neff> <decode.neff> <iodir> <W> <n_decode> [reps=1]
// Per rank r: every tensor of the same name is one device tensor on vnc r (a pool); the decode NEFF's tensors are
// allocated first (stacked weights / KV caches define the pool sizes; contents from <iodir>/r<r>/<name>.bin, else
// zeros). The layer NEFF is executed once per layer l with every tensor whose pool tensor is L x its size bound to
// slice l (nrt_tensor_allocate_slice); a name X missing from the pool but present as X0..X3 (layer groups, each
// LG x size) is bound to slice l % LG of X<l / LG>. Prefill scratch tensors are pool tensors too (zeros).
// Wiring (lab 1 host/generate): every prefill output_0 (= plen) is the decode's pos buffer X; decode set A reads X
// writes Y, set B reads Y writes X. Each rank's thread queues emb, L x layer, head and n_decode steps with
// nrta_execute_schedule (collective NEFFs need one scheduling thread per rank, lab 2).
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)
using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
static bool slurp(const std::string& p, std::vector<char>& out) {
  std::ifstream f(p, std::ios::binary); if (!f) return false;
  f.seekg(0, std::ios::end); out.resize(f.tellg()); f.seekg(0); f.read(out.data(), out.size()); return true;
}
struct Pool {
  std::string dir; int vnc; std::map<std::string, nrt_tensor_t*> t; std::map<std::string, size_t> sz;
  nrt_tensor_t* get(const std::string& name, size_t size, bool load) {
    auto it = t.find(name);
    if (it != t.end()) { if (sz[name] != size) { fprintf(stderr, "%s: size %zu vs %zu\n", name.c_str(), sz[name], size); exit(1); } return it->second; }
    nrt_tensor_t* x; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, vnc, size, name.c_str(), &x));
    std::vector<char> buf;
    if (!(load && slurp(dir + "/" + name + ".bin", buf) && buf.size() == size)) buf.assign(size, 0);
    CK(nrt_tensor_write(x, buf.data(), 0, size));
    t[name] = x; sz[name] = size; return x;
  }
};
struct Ex { nrt_model_t* m; nrt_tensor_set_t* in; nrt_tensor_set_t* out; };
static std::vector<char> rd(const std::string& p) { std::vector<char> v; if (!slurp(p, v)) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); } return v; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 6) { fprintf(stderr, "usage: tpgen emb,layer,head.neff decode.neff iodir W n_decode [reps]\n"); return 2; }
  std::string pf = argv[1];
  std::vector<std::string> pfs; { size_t a = 0, b; while ((b = pf.find(',', a)) != std::string::npos) { pfs.push_back(pf.substr(a, b - a)); a = b + 1; } pfs.push_back(pf.substr(a)); }
  std::string dir = argv[3]; int W = atoi(argv[4]), n = atoi(argv[5]); int reps = argc > 6 ? atoi(argv[6]) : 1;
  std::vector<std::vector<char>> neffs = {rd(pfs[0]), rd(pfs[1]), rd(pfs[2]), rd(argv[2])};
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  struct Rank { nrt_model_t* m[4]; Pool pool; std::vector<Ex> pre; nrt_tensor_set_t *inA, *outA, *inB, *outB; };
  std::vector<Rank> R(W);
  auto t0 = clk::now();
  for (int k = 0; k < 4; k++) for (int r = 0; r < W; r++) {
    if (W == 1) CK(nrt_load(neffs[k].data(), neffs[k].size(), r, -1, &R[r].m[k]));
    else CK(nrt_load_collectives(neffs[k].data(), neffs[k].size(), r, -1, r, W, &R[r].m[k]));
  }
  printf("nrt_load 4 NEFFs x %d ranks: %.1f ms\n", W, us_since(t0) / 1e3);
  t0 = clk::now();
  std::vector<int> Ls(W);
  {
    std::vector<std::thread> th;
    for (int r = 0; r < W; r++) th.emplace_back([&, r] {
      auto& k = R[r]; k.pool.dir = dir + "/r" + std::to_string(r); k.pool.vnc = r;
      CK(nrt_allocate_tensor_set(&k.inA)); CK(nrt_allocate_tensor_set(&k.outA));
      CK(nrt_allocate_tensor_set(&k.inB)); CK(nrt_allocate_tensor_set(&k.outB));
      nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(k.m[3], &info));
      for (uint64_t i = 0; i < info->tensor_count; i++) {
        auto& ti = info->tensor_array[i]; bool in = ti.usage == NRT_TENSOR_USAGE_INPUT; std::string nm = ti.name;
        if (nm == "pos") {
          CK(nrt_add_tensor_to_tensor_set(k.inA, ti.name, k.pool.get("posX", ti.size, false)));
          CK(nrt_add_tensor_to_tensor_set(k.inB, ti.name, k.pool.get("posY", ti.size, false)));
        } else if (nm == "output_0") {
          CK(nrt_add_tensor_to_tensor_set(k.outA, ti.name, k.pool.get("posY", ti.size, false)));
          CK(nrt_add_tensor_to_tensor_set(k.outB, ti.name, k.pool.get("posX", ti.size, false)));
        } else {
          auto x = k.pool.get(nm, ti.size, in);
          CK(nrt_add_tensor_to_tensor_set(in ? k.inA : k.outA, ti.name, x));
          CK(nrt_add_tensor_to_tensor_set(in ? k.inB : k.outB, ti.name, x));
        }
      }
      auto bind = [&](nrt_model_t* m, int layer, int L) {
        Ex e{m, nullptr, nullptr}; CK(nrt_allocate_tensor_set(&e.in)); CK(nrt_allocate_tensor_set(&e.out));
        nrt_tensor_info_array_t* inf; CK(nrt_get_model_tensor_info(m, &inf));
        for (uint64_t i = 0; i < inf->tensor_count; i++) {
          auto& ti = inf->tensor_array[i]; bool in = ti.usage == NRT_TENSOR_USAGE_INPUT;
          std::string nm = ti.name; if (nm == "output_0") nm = "posX";
          nrt_tensor_t* x;
          int lay = layer < 0 ? 0 : layer;   // emb / head NEFFs carry the per-layer tensors too (unused): slice 0
          if (layer < 0 && k.pool.t.count(nm) && k.pool.sz[nm] != ti.size && k.pool.sz[nm] % ti.size == 0) {
            CK(nrt_tensor_allocate_slice(k.pool.t[nm], 0, ti.size, ti.name, &x));
          } else if (layer < 0 && !k.pool.t.count(nm) && k.pool.t.count(nm + "0")) {
            CK(nrt_tensor_allocate_slice(k.pool.t[nm + "0"], 0, ti.size, ti.name, &x));
          } else if (layer >= 0 && k.pool.t.count(nm) && k.pool.sz[nm] == (size_t)L * ti.size) {
            CK(nrt_tensor_allocate_slice(k.pool.t[nm], (size_t)layer * ti.size, ti.size, ti.name, &x));
          } else if (layer >= 0 && !k.pool.t.count(nm) && k.pool.t.count(nm + "0")) {
            size_t LG = k.pool.sz[nm + "0"] / ti.size;
            std::string g = nm + std::to_string(lay / LG);
            CK(nrt_tensor_allocate_slice(k.pool.t[g], (size_t)(lay % LG) * ti.size, ti.size, ti.name, &x));
          } else x = k.pool.get(nm, ti.size, in);
          CK(nrt_add_tensor_to_tensor_set(in ? e.in : e.out, ti.name, x));
        }
        k.pre.push_back(e);
      };
      int L = 0;
      nrt_tensor_info_array_t* inf; CK(nrt_get_model_tensor_info(k.m[1], &inf));
      for (uint64_t i = 0; i < inf->tensor_count; i++)
        if (!strcmp(inf->tensor_array[i].name, "wqkv")) L = k.pool.sz["wqkv"] / inf->tensor_array[i].size;
      Ls[r] = L;
      bind(k.m[0], -1, 0);
      for (int l = 0; l < L; l++) bind(k.m[1], l, L);
      bind(k.m[2], -1, 0);
    });
    for (auto& x : th) x.join();
  }
  size_t tot = 0; for (auto& kv : R[0].pool.sz) tot += kv.second;
  printf("layered prefill: %d layers; per rank %zu tensors, %.2f GB, upload %.1f ms\n", Ls[0], R[0].pool.t.size(), tot / 1e9, us_since(t0) / 1e3);
  for (int rep = 0; rep < reps; rep++) {
    double tp = 0, tt = 0;
    for (int phase = 0; phase < 2; phase++) {   // 0: prefill alone; 1: prefill + n decode steps
      t0 = clk::now();
      std::vector<std::thread> th;
      for (int r = 0; r < W; r++) th.emplace_back([&, r, phase] {
        auto& k = R[r]; nrta_seq_t seq = 0; NRT_STATUS s;
        std::vector<NRT_STATUS> rets(k.pre.size() + n, NRT_SUCCESS);
        for (size_t j = 0; j < k.pre.size(); j++) { while ((s = nrta_execute_schedule(k.pre[j].m, k.pre[j].in, k.pre[j].out, 0, &rets[j], &seq)) == NRT_QUEUE_FULL) {} CK(s); }
        if (phase == 1) for (int i = 0; i < n; i++) {
          while ((s = nrta_execute_schedule(k.m[3], i % 2 ? k.inB : k.inA, i % 2 ? k.outB : k.outA, 0, &rets[k.pre.size() + i], &seq)) == NRT_QUEUE_FULL) {}
          CK(s);
        }
        bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
        for (auto x : rets) if (x != NRT_SUCCESS) { fprintf(stderr, "rank %d exec ret %d (%s)\n", r, (int)x, nrt_get_status_as_str(x)); exit(1); }
      });
      for (auto& x : th) x.join();
      (phase == 0 ? tp : tt) = us_since(t0);
    }
    printf("GEN rep %d: prefill alone (%zu executions) %.2f ms; prefill + %d decode steps %.2f ms -> decode %.1f us/step\n",
           rep, R[0].pre.size(), tp / 1e3, n, tt / 1e3, n ? (tt - tp) / n : 0.0);
  }
  for (int r = 0; r < W; r++) {
    auto& p = R[r].pool; size_t ts = p.sz["toks"]; std::vector<char> buf(ts);
    CK(nrt_tensor_read(p.t["toks"], buf.data(), 0, ts));
    std::ofstream f(p.dir + "/out_toks.bin", std::ios::binary); f.write(buf.data(), ts);
  }
  nrt_close();
  return 0;
}
