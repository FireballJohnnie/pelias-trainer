# Citations and acknowledgements (trainer)

**Status: draft, private repository.** Each entry below was checked on **2026-10-20** by opening the page and reading what it says (title, date or licence as shown). A "role" says how the source relates to this repository; an entry is listed as imported or used only where the code shows it. Where a role depends on something only the author can say, it is marked **author-stated**.

| Role | Meaning |
|---|---|
| **Imported** | Imported by the trainer's files (found by searching their text for import lines). |
| **Optional** | Looked for by the preflight script; speeds up one model part if present. Not required. |
| **Used by the export step** | Run by the export script to make the GGUF files. |
| **Base model** | The checkpoint the training starts from. |
| **Consulted** | Read while designing; the trainer's code does not import it (**author-stated**). |
| **Development tool** | Software used to help write and review the work. |

Licences shown are the ones the project page displayed on 2026-10-20. They are not a legal review; see README section 7.

---

## 1. Imported

* **PyTorch**. <https://github.com/pytorch/pytorch>. The page fetched did not name its licence, so the licence is **still to be checked**. The grouped expert matmul uses `torch.nn.functional.grouped_mm`, falling back to `torch._grouped_mm` (the 0.2 package README).
* **Transformers**, Hugging Face. <https://github.com/huggingface/transformers>. Apache-2.0. The loader imports its Qwen3.5/3.6 MoE modelling code (`transformers.models.qwen3_5_moe`).
* **PEFT**, Hugging Face. <https://github.com/huggingface/peft>. Apache-2.0. LoRA.
* **Triton**. <https://github.com/triton-lang/triton>. MIT. "Development repository for the Triton language and compiler"; the fused dequantisation kernel.
* **safetensors**. <https://github.com/safetensors/safetensors>. Apache-2.0. Reads the base shards and writes the merged shards.
* **NumPy**. <https://github.com/numpy/numpy>. The README lists it as imported; the page fetched did not name its licence, so the licence is **still to be checked**.

## 2. Optional

* **flash-linear-attention** (`fla`), fla-org. <https://github.com/fla-org/flash-linear-attention>. MIT. Includes a Gated DeltaNet implementation.
* **causal-conv1d**, Dao-AILab. <https://github.com/Dao-AILab/causal-conv1d>. BSD-3-Clause. "Causal depthwise conv1d in CUDA, with a PyTorch interface".

The 0.1 README says these two speed up the DeltaNet layers if they can be imported.

## 3. Used by the export step

* **llama.cpp and ggml**, Georgi Gerganov and contributors. <https://github.com/ggml-org/llama.cpp>. MIT licence (README section 7). `convert_hf_to_gguf.py` and `llama-quantize` make the GGUF files.
* **GGUF file format specification**, ggml-org. <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.
* **k-quants**, Iwan Kawrakow (`ikawrakow`), llama.cpp pull request #1684, merged 2023-06-05. <https://github.com/ggml-org/llama.cpp/pull/1684>. The `Q4_K_M` and `Q5_K_M` file types in `benchmarks/RESULTS.md` come from it.

## 4. Base model

* **Qwen3.6-35B-A3B**, Qwen Team. <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>. Apache-2.0. 40 layers, 256 experts, 8 routed plus 1 shared expert active per token, expert intermediate size 512, hidden size 2048, layout 10 x (3 x (Gated DeltaNet to MoE) then 1 x (Gated Attention to MoE)), as the card states.
* **The upload the export script reads**: `unsloth/Qwen3.6-35B-A3B`, Unsloth AI. <https://huggingface.co/unsloth/Qwen3.6-35B-A3B>. Apache-2.0. Its model tree lists the Qwen release as its base, so it is a derivative upload of the original.

### Base models of the author's other private models (not part of this repository)

* **Gemma 4 26B A4B**, as uploaded by Unsloth AI. <https://huggingface.co/unsloth/gemma-4-26B-A4B>. The card shows Apache-2.0, lists `google/gemma-4-26B-A4B` as its base, and its MoE table gives 25.2B total and 3.8B active parameters. The 12B variant and the exact Qwen3-Coder model are **not yet confirmed**, so they have no entry here.

## 5. Consulted (author-stated)

* **Unsloth**. <https://github.com/unslothai/unsloth>. The core package is Apache-2.0; optional components such as its Studio UI are AGPL-3.0. Its page *Fine-tune MoE Models 12x Faster with Unsloth* (<https://unsloth.ai/docs/basics/faster-moe-training>) describes MoE training backends set by `UNSLOTH_MOE_BACKEND` (`grouped_mm`, `unsloth_triton`, `native_torch`). The author's earlier private fine-tuning environment used Unsloth (**author-stated**). This repository's code does **not** import it, and the 0.1 README lists it under "Not needed"; its inclusion here is as a source read, not a dependency.

## 6. Platform documentation

* **Advanced settings configuration in WSL**, Microsoft Learn. <https://learn.microsoft.com/en-us/windows/wsl/wsl-config>. The `.wslconfig` keys `memory`, `processors`, `swap` and `swapFile`.
* **CUDA on WSL User Guide**, NVIDIA. <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>.

## 7. Development tools

* **Claude** (Anthropic): Sonnet, Opus and Fable models, used for coding, review, analysis and documentation. Exact versions are not recorded.
* **Gemini Pro and Gemini 4 Argon** (Google), used for the same kinds of work.
  *Gemini 4 Argon: our next era of frontier intelligence*, Google, 2026-09-30. <https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/>.
  *Gemini 4 Argon is now available for everyone* (headline: "Gemini 4 Argon: Built to solve your hardest problems"), Google, 2026-10-13: available in the Gemini app to Google AI Pro and Google AI Ultra subscribers and in the API from that date. <https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-argon/>.
  API reference: model ID `gemini-4-argon`, context window 1,048,576 tokens. <https://ai.google.dev/gemini-api/docs/models/gemini-4-argon>.

Which tool helped with which part is not recorded, so no per-file attribution is made. The research, the design and the training experiments are the author's.

## 8. Not cited, on purpose

FlashAttention, FlexAttention, xFormers, bitsandbytes and TRL are **not** cited: the trainer's files do not import them (the 0.1 README lists bitsandbytes, flash-attn and trl as "Not needed", and the preflight script only prints the FlashAttention version if one is installed).
