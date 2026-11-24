// tpchain: tensor-parallel decode chain. One SPMD decode-step NEFF on W logical NeuronCores (rank r on visible vnc
// first + r*stride), n_steps executions per rank, all queued, no host work between steps.
//   NEURON_RT_ROOT_COMM_ID=localhost:<port> tpchain <neff> <iodir> <W> <n_steps> <in:out>[,..] [reps=1] [first=0] [stride=1]
// Per rank: inputs from <iodir>/r<r>/<name>.bin (else zeros), every tensor allocated once on that rank's core. For each
// swap pair in:out (pos:output_0) two tensor sets are built (A reads X writes Y, B the reverse), executions alternate
// A/B. Each rank's executions are scheduled from its own thread (nrta_execute_schedule of a collective NEFF blocks
// until the peers have scheduled theirs; scheduling all ranks from one thread deadlocks). After rep 0 every non-weight
// tensor is dumped to <iodir>/r<r>/out_<name>.bin. reps>1 rewrites the small inputs + caches and repeats for timing.
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
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
struct T { std::string name; size_t size; bool in; nrt_tensor_t* t; std::vector<char> init; };
struct Rank { nrt_model_t* m = nullptr; nrt_tensor_set_t *inA, *outA, *inB, *outB; std::vector<T> ts; int vnc; };

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 6) { fprintf(stderr, "usage: tpchain neff iodir W n_steps in:out[,..] [reps] [first] [stride]\n"); return 2; }
  std::string dir = argv[2];
  int W = atoi(argv[3]), n = atoi(argv[4]);
  std::map<std::string, std::string> sw;
  { std::stringstream ss(argv[5]); std::string p;
    while (std::getline(ss, p, ',')) { auto c = p.find(':'); sw[p.substr(0, c)] = p.substr(c + 1); sw[p.substr(c + 1)] = p.substr(0, c); } }
  int reps = argc > 6 ? atoi(argv[6]) : 1;
  int first = argc > 7 ? atoi(argv[7]) : 0, stride = argc > 8 ? atoi(argv[8]) : 1;
  if (W > 1 && !getenv("NEURON_RT_ROOT_COMM_ID")) { fprintf(stderr, "set NEURON_RT_ROOT_COMM_ID\n"); return 2; }
  std::vector<char> neff;
  if (!slurp(argv[1], neff)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  std::vector<Rank> R(W);
  auto t0 = clk::now();
  for (int r = 0; r < W; r++) {
    R[r].vnc = first + r * stride;
    if (W == 1) CK(nrt_load(neff.data(), neff.size(), R[r].vnc, -1, &R[r].m));
    else CK(nrt_load_collectives(neff.data(), neff.size(), R[r].vnc, -1, r, W, &R[r].m));
  }
  printf("nrt_load x %d: %.1f ms (%zu bytes)\n", W, us_since(t0) / 1e3, neff.size());
  t0 = clk::now();
  {
    std::vector<std::thread> th;
    for (int r = 0; r < W; r++) th.emplace_back([&, r] {
      auto& k = R[r];
      nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(k.m, &info));
      CK(nrt_allocate_tensor_set(&k.inA)); CK(nrt_allocate_tensor_set(&k.outA));
      CK(nrt_allocate_tensor_set(&k.inB)); CK(nrt_allocate_tensor_set(&k.outB));
      std::map<std::string, nrt_tensor_t*> byname;
      for (uint64_t i = 0; i < info->tensor_count; i++) {
        auto& ti = info->tensor_array[i];
        T t{ti.name, ti.size, ti.usage == NRT_TENSOR_USAGE_INPUT, nullptr, {}};
        CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, t.size, ti.name, &t.t));
        std::vector<char> buf;
        std::string f = dir + "/r" + std::to_string(r) + "/" + t.name + ".bin";
        if (t.in && slurp(f, buf)) {
          if (buf.size() != t.size) { fprintf(stderr, "%s: file %zu bytes, tensor %zu\n", f.c_str(), buf.size(), t.size); exit(1); }
        } else buf.assign(t.size, 0);
        CK(nrt_tensor_write(t.t, buf.data(), 0, t.size));
        if (t.in && (t.size < (64u << 20) || t.name.find("cache") != std::string::npos)) t.init = std::move(buf);
        byname[t.name] = t.t;
        k.ts.push_back(std::move(t));
      }
      for (auto& t : k.ts) {
        if (!sw.count(t.name)) {
          CK(nrt_add_tensor_to_tensor_set(t.in ? k.inA : k.outA, t.name.c_str(), t.t));
          CK(nrt_add_tensor_to_tensor_set(t.in ? k.inB : k.outB, t.name.c_str(), t.t));
        } else {
          CK(nrt_add_tensor_to_tensor_set(t.in ? k.inA : k.outA, t.name.c_str(), t.t));
          CK(nrt_add_tensor_to_tensor_set(t.in ? k.inB : k.outB, t.name.c_str(), byname[sw[t.name]]));
        }
      }
    });
    for (auto& t : th) t.join();
  }
  size_t tot = 0; for (auto& t : R[0].ts) tot += t.size;
  printf("upload %.1f ms (%.2f GB per rank)\n", us_since(t0) / 1e3, tot / 1e9);
  for (int rep = 0; rep < reps; rep++) {
    if (rep > 0) for (auto& k : R) for (auto& t : k.ts) if (!t.init.empty()) CK(nrt_tensor_write(t.t, t.init.data(), 0, t.size));
    t0 = clk::now();
    std::vector<std::thread> th;
    for (int r = 0; r < W; r++) th.emplace_back([&, r] {
      auto& k = R[r];
      std::vector<NRT_STATUS> rets(n); nrta_seq_t seq = 0;
      for (int i = 0; i < n; i++) {
        NRT_STATUS s;
        while ((s = nrta_execute_schedule(k.m, i % 2 ? k.inB : k.inA, i % 2 ? k.outB : k.outA, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {}
        CK(s);
      }
      bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
      for (auto x : rets) if (x != NRT_SUCCESS) { fprintf(stderr, "rank %d exec ret %d (%s)\n", r, (int)x, nrt_get_status_as_str(x)); exit(1); }
    });
    for (auto& t : th) t.join();
    double us = us_since(t0);
    printf("CHAIN rep %d: %d steps x %d ranks in %.1f us = %.2f us/step\n", rep, n, W, us, us / n);
    if (rep == 0) for (int r = 0; r < W; r++) for (auto& t : R[r].ts) {
      if (t.name[0] == 'w' || t.name == "embed") continue;
      std::vector<char> buf(t.size); CK(nrt_tensor_read(t.t, buf.data(), 0, t.size));
      std::ofstream f(dir + "/r" + std::to_string(r) + "/out_" + t.name + ".bin", std::ios::binary); f.write(buf.data(), t.size);
    }
  }
  for (auto& k : R) nrt_unload(k.m);
  nrt_close();
  return 0;
}
