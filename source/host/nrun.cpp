// nrun: generic libnrt NEFF runner used by every self-check in the lab.
//   nrun <model.neff> <iodir> [n_exec=1] [mode=sync|queue] [bench_iters=0]
// Inputs: <iodir>/<name>.bin if present (size must match), else zero-filled. All tensors are allocated
// once on the device; the NEFF is executed n_exec times back to back with the same tensor sets (so
// kernels that update inputs in place, e.g. a KV cache + position, advance step by step on the device).
// Afterwards every input and output is read back to <iodir>/out_<name>.bin.
// bench_iters>0: additionally times bench_iters queued executions (nrta_execute_schedule, all in flight)
// and bench_iters one-at-a-time sync executions, AFTER the dump (state keeps advancing).
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
#include <vector>
#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)
using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
static bool slurp(const std::string& p, std::vector<char>& out) {
  std::ifstream f(p, std::ios::binary); if (!f) return false;
  out.assign((std::istreambuf_iterator<char>(f)), {}); return true;
}
int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: nrun model.neff iodir [n_exec] [sync|queue] [bench_iters]\n"); return 2; }
  std::string dir = argv[2];
  int n_exec = argc > 3 ? atoi(argv[3]) : 1;
  bool queue = argc > 4 && !strcmp(argv[4], "queue");
  int bench = argc > 5 ? atoi(argv[5]) : 0;
  std::vector<char> neff;
  if (!slurp(argv[1], neff)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  nrt_model_t* m; auto t0 = clk::now();
  CK(nrt_load(neff.data(), neff.size(), 0, 1, &m));
  printf("nrt_load %.1f ms (%zu bytes)\n", us_since(t0) / 1e3, neff.size());
  nrt_tensor_info_array_t* info; CK(nrt_get_model_tensor_info(m, &info));
  nrt_tensor_set_t *ins, *outs; CK(nrt_allocate_tensor_set(&ins)); CK(nrt_allocate_tensor_set(&outs));
  struct T { std::string name; size_t size; bool in; nrt_tensor_t* t; };
  std::vector<T> ts;
  for (uint64_t i = 0; i < info->tensor_count; i++) {
    auto& ti = info->tensor_array[i];
    T t{ti.name, ti.size, ti.usage == NRT_TENSOR_USAGE_INPUT, nullptr};
    CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, t.size, ti.name, &t.t));
    std::vector<char> buf;
    if (t.in && slurp(dir + "/" + t.name + ".bin", buf)) {
      if (buf.size() != t.size) { fprintf(stderr, "%s: file %zu bytes, tensor %zu\n", t.name.c_str(), buf.size(), t.size); return 1; }
    } else buf.assign(t.size, 0);
    CK(nrt_tensor_write(t.t, buf.data(), 0, t.size));
    CK(nrt_add_tensor_to_tensor_set(t.in ? ins : outs, ti.name, t.t));
    printf("  %-6s %-24s %zu bytes\n", t.in ? "in" : "out", t.name.c_str(), t.size);
    ts.push_back(t);
  }
  auto run_queued = [&](int n) {
    std::vector<NRT_STATUS> rets(n); nrta_seq_t seq = 0;
    for (int i = 0; i < n; i++) { NRT_STATUS s; while ((s = nrta_execute_schedule(m, ins, outs, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {} CK(s); }
    bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
    for (auto r : rets) if (r != NRT_SUCCESS) { fprintf(stderr, "exec ret %d (%s)\n", (int)r, nrt_get_status_as_str(r)); exit(1); }
  };
  t0 = clk::now();
  if (queue) run_queued(n_exec); else for (int i = 0; i < n_exec; i++) CK(nrt_execute(m, ins, outs));
  printf("executed %d x (%s) in %.1f us total\n", n_exec, queue ? "queued" : "sync", us_since(t0));
  for (auto& t : ts) {
    std::vector<char> buf(t.size); CK(nrt_tensor_read(t.t, buf.data(), 0, t.size));
    std::ofstream f(dir + "/out_" + t.name + ".bin", std::ios::binary); f.write(buf.data(), t.size);
  }
  if (bench > 0) {
    run_queued(10);
    t0 = clk::now(); run_queued(bench);
    printf("BENCH queued: %.2f us/exec over %d\n", us_since(t0) / bench, bench);
    std::vector<double> l;
    for (int i = 0; i < bench; i++) { auto t1 = clk::now(); CK(nrt_execute(m, ins, outs)); l.push_back(us_since(t1)); }
    std::sort(l.begin(), l.end());
    printf("BENCH sync nrt_execute: p50 %.2f us p10 %.2f p90 %.2f\n", l[l.size() / 2], l[l.size() / 10], l[l.size() * 9 / 10]);
  }
  nrt_close();
  return 0;
}
