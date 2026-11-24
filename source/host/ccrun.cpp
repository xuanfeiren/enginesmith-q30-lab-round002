// ccrun: run one collective NEFF (compiled for W ranks with nki.collectives) on W logical NeuronCores from ONE
// process, natively on libnrt.
//   NEURON_RT_ROOT_COMM_ID=localhost:<port> NEURON_RT_VISIBLE_CORES=56-59 ccrun <neff> <iodir> <W> [n_exec=1] [bench=0] [first_vnc=0] [vnc_stride=1]
// Rank r is loaded with nrt_load_collectives(neff, vnc = first_vnc + r*stride, ctx_device_id = r, ctx_device_count = W)
// (all loads in parallel threads: the loads rendezvous). Inputs per rank: <iodir>/r<r>/<name>.bin (else zeros).
// n_exec executions are queued on every rank (nrta_execute_schedule, all in flight), then every tensor is dumped to
// <iodir>/r<r>/out_<name>.bin. bench>0: then times bench queued executions per rank (all ranks in flight).
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)
using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
static bool slurp(const std::string& p, std::vector<char>& out) {
  std::ifstream f(p, std::ios::binary); if (!f) return false;
  out.assign((std::istreambuf_iterator<char>(f)), {}); return true;
}
struct T { std::string name; size_t size; bool in; nrt_tensor_t* t; };
struct Rank { nrt_model_t* m = nullptr; nrt_tensor_set_t *ins, *outs; std::vector<T> ts; int vnc; };

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: ccrun neff iodir W [n_exec] [bench] [first_vnc] [stride]\n"); return 2; }
  if (!getenv("NEURON_RT_ROOT_COMM_ID")) { fprintf(stderr, "set NEURON_RT_ROOT_COMM_ID=localhost:<port>\n"); return 2; }
  std::string dir = argv[2];
  int poll_us = getenv("CC_POLL_US") ? atoi(getenv("CC_POLL_US")) : 0;   // sleep between completion polls
  int W = atoi(argv[3]);
  int n_exec = argc > 4 ? atoi(argv[4]) : 1;
  int bench = argc > 5 ? atoi(argv[5]) : 0;
  int first = argc > 6 ? atoi(argv[6]) : 0, stride = argc > 7 ? atoi(argv[7]) : 1;
  std::vector<char> neff;
  if (!slurp(argv[1], neff)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  setvbuf(stdout, nullptr, _IONBF, 0);
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  std::vector<Rank> R(W);
  auto t0 = clk::now();
  if (getenv("CC_PAR_LOAD")) {
    std::vector<std::thread> th;
    for (int r = 0; r < W; r++) th.emplace_back([&, r] {
      R[r].vnc = first + r * stride;
      CK(nrt_load_collectives(neff.data(), neff.size(), R[r].vnc, -1, r, W, &R[r].m));
    });
    for (auto& t : th) t.join();
  } else for (int r = 0; r < W; r++) {   // sequential, as nki's own multi-rank runner does
    R[r].vnc = first + r * stride;
    if (W == 1) CK(nrt_load(neff.data(), neff.size(), R[r].vnc, -1, &R[r].m));
    else CK(nrt_load_collectives(neff.data(), neff.size(), R[r].vnc, -1, r, W, &R[r].m));
    fprintf(stderr, "loaded rank %d on vnc %d\n", r, R[r].vnc);
  }
  printf("nrt_load_collectives x %d: %.1f ms\n", W, us_since(t0) / 1e3);
  t0 = clk::now();
  {
    std::vector<std::thread> th;   // allocate + upload every rank's tensors in parallel (one thread per rank)
    for (int r = 0; r < W; r++) th.emplace_back([&, r] {
      auto& k = R[r];
      nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(k.m, &info));
      CK(nrt_allocate_tensor_set(&k.ins)); CK(nrt_allocate_tensor_set(&k.outs));
      for (uint64_t i = 0; i < info->tensor_count; i++) {
        auto& ti = info->tensor_array[i];
        T t{ti.name, ti.size, ti.usage == NRT_TENSOR_USAGE_INPUT, nullptr};
        CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, k.vnc, t.size, ti.name, &t.t));
        std::vector<char> buf;
        std::string f = dir + "/r" + std::to_string(r) + "/" + t.name + ".bin";
        if (t.in && slurp(f, buf)) {
          if (buf.size() != t.size) { fprintf(stderr, "%s: file %zu bytes, tensor %zu\n", f.c_str(), buf.size(), t.size); exit(1); }
        } else buf.assign(t.size, 0);
        CK(nrt_tensor_write(t.t, buf.data(), 0, t.size));
        CK(nrt_add_tensor_to_tensor_set(t.in ? k.ins : k.outs, ti.name, t.t));
        if (r == 0) printf("  %-4s %-24s %zu bytes\n", t.in ? "in" : "out", t.name.c_str(), t.size);
        k.ts.push_back(t);
      }
    });
    for (auto& t : th) t.join();
  }
  printf("upload %.1f ms\n", us_since(t0) / 1e3);
  auto run_sync = [&](int n) {   // one thread per rank, nrt_execute n times
    std::vector<std::thread> th;
    for (int r = 0; r < W; r++) th.emplace_back([&, r] { for (int i = 0; i < n; i++) CK(nrt_execute(R[r].m, R[r].ins, R[r].outs)); });
    for (auto& t : th) t.join();
  };
  bool sync = getenv("CC_SYNC") != nullptr;
  auto run_queued0 = [&](int n) {
    std::vector<std::vector<NRT_STATUS>> rets(W, std::vector<NRT_STATUS>(n));
    std::vector<nrta_seq_t> seq(W, 0);
    for (int i = 0; i < n; i++)
      for (int r = 0; r < W; r++) {
        NRT_STATUS s;
        while ((s = nrta_execute_schedule(R[r].m, R[r].ins, R[r].outs, 0, &rets[r][i], &seq[r])) == NRT_QUEUE_FULL) {}
        CK(s);
      }
    for (int r = 0; r < W; r++) { bool done = false; while (!done) CK(nrta_is_completed(seq[r], &done)); }
    for (int r = 0; r < W; r++) for (auto x : rets[r]) if (x != NRT_SUCCESS) { fprintf(stderr, "rank %d exec ret %d (%s)\n", r, (int)x, nrt_get_status_as_str(x)); exit(1); }
  };
  auto run_queued_mt = [&](int n) {   // one scheduling thread per rank
    std::vector<std::thread> th;
    for (int r = 0; r < W; r++) th.emplace_back([&, r] {
      std::vector<NRT_STATUS> rets(n); nrta_seq_t seq = 0;
      for (int i = 0; i < n; i++) { NRT_STATUS s; while ((s = nrta_execute_schedule(R[r].m, R[r].ins, R[r].outs, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {} CK(s); }
      bool done = false; while (!done) { CK(nrta_is_completed(seq, &done)); if (poll_us > 0 && !done) usleep(poll_us); }
      for (auto x : rets) if (x != NRT_SUCCESS) { fprintf(stderr, "rank %d exec ret %d (%s)\n", r, (int)x, nrt_get_status_as_str(x)); exit(1); }
    });
    for (auto& t : th) t.join();
  };
  bool mt = W > 1 || getenv("CC_MT") != nullptr;   // one thread per rank (a single thread deadlocks, lab 2)
  auto run_queued = [&](int n) { if (sync) run_sync(n); else if (mt) run_queued_mt(n); else run_queued0(n); };
  auto dump = [&](const std::string& suf) {
    for (int r = 0; r < W; r++) for (auto& t : R[r].ts) {
      if (t.size > (256u << 20) || (t.in && !suf.empty())) continue;
      std::vector<char> buf(t.size); CK(nrt_tensor_read(t.t, buf.data(), 0, t.size));
      std::ofstream f(dir + "/r" + std::to_string(r) + "/out_" + t.name + suf + ".bin", std::ios::binary); f.write(buf.data(), t.size);
    }
  };
  int nin = getenv("CC_NIN") ? atoi(getenv("CC_NIN")) : 0;
  if (nin > 0) {   // multi-input mode: execution i first rewrites every input that has a file <name>.<i>.bin
    for (int i = 0; i < nin; i++) {
      for (int r = 0; r < W; r++) for (auto& t : R[r].ts) {
        std::vector<char> buf;
        if (t.in && slurp(dir + "/r" + std::to_string(r) + "/" + t.name + "." + std::to_string(i) + ".bin", buf)) {
          if (buf.size() != t.size) { fprintf(stderr, "input %s.%d: size mismatch\n", t.name.c_str(), i); exit(1); }
          CK(nrt_tensor_write(t.t, buf.data(), 0, t.size));
        }
      }
      run_queued(1);
      dump("." + std::to_string(i));
    }
    printf("executed %d input sets\n", nin);
  } else {
    t0 = clk::now();
    run_queued(n_exec);
    fprintf(stderr, "queued done\n"); printf("executed %d x %d ranks (queued) in %.1f us\n", n_exec, W, us_since(t0));
    dump("");
  }
  if (bench > 0) {
    run_queued(10);
    int nrep = getenv("CC_REPS") ? atoi(getenv("CC_REPS")) : 1;
    std::vector<double> v;
    for (int i = 0; i < nrep; i++) { t0 = clk::now(); run_queued(bench); v.push_back(us_since(t0) / bench); }
    std::sort(v.begin(), v.end());
    printf("BENCH queued: %.2f us/exec over %d (all %d ranks in flight; min of %d reps, median %.2f, max %.2f)\n", v[0], bench, W, nrep, v[v.size() / 2], v.back());
  }
  for (auto& k : R) nrt_unload(k.m);
  nrt_close();
  return 0;
}
