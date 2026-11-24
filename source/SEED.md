# Seed: kernel lab 3 (mixture of experts)

The work of kernel lab 3, not yet an engine: FINDINGS.md has measured, tested NKI kernels for Qwen3-30B-A3B over 4
logical NeuronCores: the MoE decode step (router, top-8 with the reference\x27s rules, gathered experts, combine), the
whole 48-layer decode step (4.97 ms/step at TP=4, identical to HF fp32 on real tokens) and MoE prefill (68.9 / 113.6 /
319.2 ms at 256 / 1,024 / 4,096 tokens), with C++ drivers on libnrt. Read FINDINGS.md first.

To turn it into the engine: a native executable that speaks the EngineSmith protocol (build.sh -> target/engine) and
runs tensor parallel 4 over the first four visible NeuronCores (NEURON_RT_VISIBLE_CORES lists eight; the last four are
the reference judge\x27s), every NEFF embedded, built only through NKI\x27s documented route. Never commit target/ or built
NEFFs. Experiments use NeuronCores 8-15.
