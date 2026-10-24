# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# kern: All device code. Depends only on hal and plain structs of pointers and shapes; never includes model.
# pelias-deps: hal
pelias_module(kern
    CU gemm_gguf/dequant_ref.cu gemm_gguf/mma_bf16.cu gemm_gguf/mma_int8.cu gemm_gguf/grouped_tiles.cu gemm_gguf/dgrad_t.cu route/router_topk.cu route/tile_map.cu route/combine.cu elementwise/rmsnorm.cu elementwise/swiglu.cu elementwise/gated_norm.cu elementwise/residual.cu elementwise/cast.cu attn/attn_fwd.cu attn/attn_bwd.cu attn/attn_ref_blocked.cu attn/rope.cu deltanet/conv1d.cu deltanet/gates.cu deltanet/l2norm.cu deltanet/chunk_prep.cu deltanet/chunk_state.cu deltanet/chunk_out.cu deltanet/chunk_bwd.cu deltanet/seq_ref.cu loss/ce_chunk.cu optim/adamw.cu optim/grad_norm.cu optim/lora_grad.cu
    DEPS hal)
