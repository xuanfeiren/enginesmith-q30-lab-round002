# Third-party code

| project | files | license | how used |
|---|---|---|---|
| AWS NKI Library (`nkilib`, github.com/aws-neuron/nki-library), the copy shipped in the neuronx-cc 2.27.5334 wheel | `nkilib/core/attention/attention_cte.py` (and its nkilib imports) | Apache-2.0 (Copyright Amazon.com, Inc. or its affiliates) | imported at build time by `kernels/prefill_tp.py` (prefill attention `_attention_cte`); compiled into the prefill layer NEFFs. Not copied into this repository; nothing of it runs on the host. |
| nkilib `selective_expert_impl.py` | pattern only (expert slab DMA with an SBUF scalar offset), re-implemented in `kernels/moe_tp.py` | Apache-2.0 | no code copied |
