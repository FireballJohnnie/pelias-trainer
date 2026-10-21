# Export record: fine-tuned Qwen3.6-35B-A3B to GGUF (2026-10-19)

A record of what was produced, how, and with which checksums. It claims nothing about quality: no server has loaded these files yet (see section 5).

## 1. Files

| File | Bytes | Size | Bits per weight | Quantize time | sha256 |
|---|---|---|---|---|---|
| `test-model-bf16.gguf` (kept on D: for the KL/perplexity comparison, delete when done) | 71,066,995,008 | 66.2 GiB (71.1 GB), 753 tensors | 16 | n/a | not recorded |
| `test-model-Q5_K_M.gguf` | 25,347,532,800 | 23.6 GiB | 5.71 | 703.7 s | `ccc6720d5f19bcad6e21e4a1825c708c7e0441d3caa5b062aa6dbd96af2f46f0` |
| `test-model-Q4_K_M.gguf` | 21,713,463,296 | 20.2 GiB | 4.89 | 735.3 s | `326e54aae9cb313b7469c7c210857c03272ce080c7d1215556faeb1123f7d3fe` |

- The quantizer reported 67,764.29 MiB of source tensors (16.01 bits per weight) and quantized sizes of 24,162.81 MiB (Q5_K_M) and 20,697.09 MiB (Q4_K_M). The 4-bit file is about 14.3% (about 3.4 GiB) smaller than the 5-bit one.
- The Q5_K_M hash was written by the export step when it finished. The Q4_K_M hash was computed afterwards from the file on disk (1 min 46 s). Each is stored next to its file as `<name>.gguf.sha256`.
- Check both: `cd <gguf-folder> && sha256sum -c test-model-Q4_K_M.gguf.sha256 test-model-Q5_K_M.gguf.sha256`

## 2. How they were made

1. Start from the original bf16 base shards and the phase-2 LoRA adapter (rank 16, alpha 32, 310 modules, 21,166,080 trainable parameters).
2. Merge: `W' = W + (alpha/r) * B @ A` with scale 2, shard by shard. 310 of 310 modules merged.
3. Convert: llama.cpp `convert_hf_to_gguf.py --outtype bf16` (llama.cpp commit 2ca15f540). 753 tensors. The chat template check passed (`tool_call`, `reasoning_content` and `enable_thinking` present in the template).
4. Quantize: `llama-quantize` (CUDA build under `build-wsl`) from that one bf16 file. Q5_K_M came from the launcher's export step; Q4_K_M was run by hand with the same tool and source file.
5. Importance matrix: none for Q4_K_M (no `--imatrix` was passed). For Q5_K_M the launcher computes one only when run with `--imatrix`, and the quantize started right after the bf16 file was finished (file times), so none was computed; that is inferred from timing, not from a log line. Published community 4-bit builds (Unsloth "UD") do use one, so scores may not transfer exactly.

Environment: Windows 11 with WSL2 Ubuntu 26.04, GCC 15.2.0, CUDA toolkit 13.0.88 for the llama.cpp build, torch 2.14.1+cu130, transformers 5.17.0, RTX 4070 12 GB, 64 GB RAM (WSL limited to 48 GB).

## 3. What the Q4_K_M mix contains

Q4_K_M is not uniformly 4-bit. The first tensors of its log were checked against the quantizer rules in the local llama.cpp source (`src/llama-quant.cpp`) and the sizes were checked by arithmetic: the output layer is 6-bit (q6_K); the token embedding and most weights are 4-bit (q4_K); the first and last eighth of the layers, and every third layer between, get 6-bit for the feed-forward down projection and the attention-value-like tensors. Only the first nine tensors of the log were examined.

## 4. Quirks seen during export

- The converter warned "Unknown RoPE type: default". Read from the converter source, this is cosmetic (the result is no rope scaling, theta 1e7).
- The GGUF includes one extra block (block 40): the base model's multi-token-prediction head. Read from the source, the loader skips it unless asked, so the layer count should read 40. This is not yet confirmed at run time.

## 5. Not yet verified

- Server load log (40 layers, no tensor errors, GPU and RAM use).
- Output quality, speed (tokens/s) and the format, tool and long-context checks, for each file.
- The bf16 file's hash.
- Whether 4-bit and 5-bit differ on this machine: that is what the tests are for.
