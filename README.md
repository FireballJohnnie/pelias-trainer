# PELIAS TRAINER

**Fine-tuning a large Mixture-of-Experts model on one 12 GB consumer GPU: the always-on part trains on the GPU, the frozen quantised experts stream in from pinned system RAM, one layer at a time.**
*Companion to [pelias-engine](https://github.com/FireballJohnnie/pelias-engine) (inference). The two meet at one interface: the GGUF file.*

> **PRE-RELEASE DRAFT, 2026-10-20. Not published.**
> This repository is private and holds a draft. This file describes what the code in the author's development folder does, as read from its own documentation and imports. **Numbers from logs are in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)** (one earlier training run with the 0.2 launcher, and the export record and tests of the 4-bit and 5-bit GGUF files, which come from a later run whose own log is not published yet); this file gives no training-speed, memory or accuracy number of its own. Sources, with their roles, are in [`CITATIONS.md`](CITATIONS.md).
> Every number in this file carries one label: **measured** (counted or run on the author's machine), **published** (vendor or standard specification), **derived** (arithmetic from labelled inputs), **estimate** (a guess, with the method to replace it), **author-stated** (the author told me; no log attached), **unmeasured** (with a planned way to measure it).
> No "first ever" claim is made.

---

## Contents
1. [What it is](#1-what-it-is)
2. [How it works](#2-how-it-works)
3. [Parts, and the status of each](#3-parts-and-the-status-of-each)
4. [Requirements](#4-requirements)
5. [Size of the code](#5-size-of-the-code)
6. [How it relates to pelias-engine](#6-how-it-relates-to-pelias-engine)
7. [Licensing and third-party notices](#7-licensing-and-third-party-notices)
8. [Before release](#8-before-release)
9. [Roadmap and open questions](#9-roadmap-and-open-questions)
10. [Acknowledgements and provenance](#10-acknowledgements-and-provenance)

---

## 1. What it is

A LoRA fine-tuning pipeline for one model family, **Qwen3.6-35B-A3B** (a Mixture-of-Experts model: 256 routed experts per layer, 8 active per token), built for a single machine with a 12 GB GPU and 64 GB of system RAM. The package's own description is: *"Spine-resident, expert-streaming LoRA trainer for Qwen3.6-35B-A3B on one 12 GB GPU."* **Status (author-stated): working, but unfinished.** **Ultimate goal (author-stated):** a universal trainer tool for better training of Mixture-of-Experts models. "Universal" is a goal, not a claim: this draft can describe only the Qwen3.6 LoRA path below. The model does not fit in 12 GB, and the usual 4-bit training route was reported by the author to be very difficult for this model (**author-stated**), so the trainer does its own expert handling instead.

It also contains a second tool: an aligner for the model's **MTP draft head** (the small extra head that proposes tokens for speculative decoding), which makes a server accept more of the tokens that head proposes.

## 2. How it works

All of this is read from the code's own documentation; none of it has been re-run for this draft.

* **Spine on the GPU.** The always-on part of the model (attention, the linear-attention layers, the shared expert) stays resident in bf16 on the GPU. **LoRA adapters are trained there, and only there.**
* **Experts frozen, quantised, in RAM.** The routed experts are quantised with an affine, group-of-64 scheme (4-bit; 8-bit optional per layer) and kept in pinned host RAM.
* **Streaming, one layer at a time.** A streamer copies each layer's experts from pinned RAM into a small pool of GPU slots, runs the layer, and moves on. The backward pass streams the weights again.
* **Expert compute.** A custom function does the expert math with an analytic SwiGLU backward. Two backends exist, a per-expert loop and a grouped matrix multiply; the faster one is chosen at start-up **only after it is checked against the reference**, with fallback to the slower path.
* **Fused dequantisation.** An optional Triton kernel unpacks and scales the weights in one pass; a plain torch path is the fallback.
* **Nothing is truncated.** Data preparation tokenises chat-format JSONL with assistant-only labels and a hard length audit (a stated requirement of the author's).
* **Low-memory merge.** The merge step folds the adapter into the base checkpoint one shard at a time, because a whole-model merge needs far more RAM than the machine has (the code's documentation says about 67 GiB; **not re-measured**).
* **v0.3 adjustments** (applied in memory by a launcher, nothing on disk is edited): a fix so the logged peak-VRAM figure covers every micro-batch of an accumulation group; a KV-cached probe that checks itself against the uncached path and falls back if they differ; and an **opt-in** offload of each layer's input to pinned RAM between forward and backward.

## 3. Parts, and the status of each

| Part | What it is | Status |
|---|---|---|
| Pelias 0.2.1 package (`pelias/`, 10 modules) | quantiser, loader, streamer, expert compute, Triton dequant, data prep, trainer, shard merge | Written 2026-10-19. Its own README says it had **not yet been executed on the target PC** when written. The author states it is now **working but unfinished** (**author-stated**). One earlier training run with the 0.2 launcher is logged in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md); that log says only "0.2"; the author states the package was 0.2.1 and that its runs were still labelled "0.2" (**author-stated**). |
| Pelias 0.1 package (15 files) and 0.1 launcher (1 script) | the same kinds of parts as 0.2 by the file lists, without 0.2's three flagged paths (8-bit resident layer 0, grouped matmul, fused Triton dequant) | Written 2026-10-19; its README and launcher say it had **not yet been run on the target PC** when written. **No benchmarks exist for it** (**author-stated**). The 0.2 files describe 0.2 as 0.1 plus three flagged paths; the author describes 0.1 as a **different path rather than an older version** (**author-stated**). |
| 0.2 launcher (1 script) | runs the stages; refuses to run as root, checks the package archive's checksum, never overwrites | Has its own built-in checks. One earlier run is logged in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) (section 2). |
| v0.3 patch layer (2 files) | the three adjustments above | Written; two on by default, one opt-in. One planned item (a "zero-sync routing plan") was **left out on purpose** after review found it could not work as drafted. Results not recorded here. |
| MTP draft-head aligner (6 files) | trains a small LoRA on the draft head against the model's own outputs, dumps the hidden states it needs, merges and exports a draft GGUF | Has a numpy reference and a torch-free test path. **No acceptance-rate result is attached.** |
| Tests (7 files, assigned by file name) | CPU-only checks, a fake torch for running the maths without torch | Not re-run for this draft. |

The aligner reports, on held-out text: **overlap** (expected acceptance when the server samples), **agree** (top-token match, i.e. greedy acceptance), and the two losses (KL, cross-entropy). Overlap is the number to watch. No value is given here.

### 3.1 The models built with it (description only)

**This repository holds no model file, no adapter and no training data** (author's decision, 2026-10-20). Parts of the author's design notes also stay private. What is described here is enough to see what was built and to rebuild an equivalent model from your own data.

**How it came about.** The work began as a training problem: fine-tuning this model on one 12 GB card. Measuring how fast the experts could be streamed from RAM during training, and which parts of the machine sat idle meanwhile, led to the inference question that became pelias-engine.

| Item | Value | Label |
|---|---|---|
| Base checkpoint | the Hugging Face `unsloth/Qwen3.6-35B-A3B` snapshot, read from the local cache by the export script | measured (script text); the licence of this exact upload is **to be checked** |
| Training | LoRA on the bf16 spine; experts frozen, group-64 affine quantised, streamed layer by layer (section 2) | read from the code |
| Training data | chat-format JSONL, tokenised with assistant-only labels and a length audit; **not published**; size in samples and tokens **unmeasured** (plan: print both from the tokenised file) | |
| Export path | per-shard merge, then a bf16 GGUF through llama.cpp's converter, then an optional importance matrix (off by default; whether it was used for these files is **unrecorded**), then one quantisation step | read from the export script |
| 4-bit build, Q4_K_M | 21,713,463,296 B (20.22 GiB), file dated 2026-10-20 01:08 | measured |
| 5-bit build, Q5_K_M | 25,347,532,800 B (23.61 GiB), file dated 2026-10-20 00:49 | measured |
| Disk to rebuild | the script checks for about 72 GB free (merge) and about 70 GB free (bf16 GGUF) | stated in the script, not re-measured |
| Trainer version that made them | the author states the published (earlier) run used the 0.2 launcher with the 0.2.1 patch, applied on the fly in another conversation and so not named in its log, which says only "0.2"; the later run that made the GGUF files is, the author states, from what will become 0.3, and its log is not published yet | **author-stated** |

Speeds of these two files under stock llama.cpp, and the modelled ceilings with the engine, are in the pelias-engine README (section 4).

## 4. Requirements

From the imports found by grep (**measured**): PyTorch, `transformers` (the Qwen3.5/3.6 MoE classes), PEFT, Triton (optional), NumPy. The grouped-matmul path prefers `torch.nn.functional.grouped_mm` (PyTorch 2.10 or newer, per the code's README) and falls back to `torch._grouped_mm`.

| Need | Value | Label |
|---|---|---|
| GPU | NVIDIA, 12 GB class | developed on one RTX 4070 12 GB (**author-stated**); other cards **untested** |
| System RAM | 64 GB on the author's machine; the quantised-expert cache size is **unmeasured** (plan: `du -sh` of the cache folder after one run) | |
| OS | Linux, native or WSL2 | developed on WSL2 Ubuntu 26.04.1 LTS; native Linux **untested** |
| Minimum versions of `transformers`, PEFT, Triton | **unmeasured** (plan: record the versions in a passing run, then bisect downwards) | |

## 5. Size of the code

Counted in the author's development folder (**measured**; bytes from file sizes).

| Part | Files | Bytes |
|---|---|---|
| 0.2.1 package archive contents (package, 5 scripts, 1 test, its README) | 17 | 102,542 |
| 0.2 launcher | 1 | 166,308 |
| 0.1 package archive contents (package, 4 scripts, 1 test, its README) | 15 | 84,381 |
| 0.1 launcher | 1 | 166,297 |
| v0.3 patch layer | 2 | 23,778 |
| MTP draft-head aligner (including a C++ hidden-state dumper) | 6 | 90,903 |
| Tests tied to the above (assigned by file name) | 7 | 90,801 |
| **Total** | **49** | **725,010 (about 708 KiB, derived)** |

The runtime environment (PyTorch, Triton, PEFT, `transformers` and their GPU kernels) is **installed by the user, never bundled**. Its size is **unmeasured** (plan: `du -sh` of the virtual environment).

## 6. How it relates to pelias-engine

* **Same idea, two directions.** Both stream quantised experts from pinned RAM to the GPU. The trainer does it layer by layer for training; the engine does it per token for inference.
* **The GGUF is the interface.** The engine runs any GGUF, trained with this trainer or not. The aligner's output is a small draft-model GGUF that llama.cpp's MTP option can load.
* **Why two repositories.** Different language and dependencies (Python / PyTorch against C++ / CUDA), a different release pace while the trainer is still being refined, and a separate licence review of its dependencies.

## 7. Licensing and third-party notices

*This is a working draft, not legal advice. The author decides, and a qualified person should review before release.*

| Item | Licence | Source |
|---|---|---|
| This repository's code | Apache-2.0 **(proposed; the author has not yet confirmed the licence or the copyright holder)** | `LICENSE` is the standard text, same as pelias-engine |
| Qwen3.6-35B-A3B (base model; weights are **not** in the repository) | Apache-2.0 | Hugging Face model page, checked 2026-10-20 |
| `transformers`, PEFT, Triton, `safetensors` (imported by the code) | Apache-2.0, Apache-2.0, MIT, Apache-2.0 | each project's GitHub page, checked 2026-10-20 |
| PyTorch, NumPy (imported by the code) | **not yet checked** | the pages fetched on 2026-10-20 did not name a licence; read each project's own `LICENSE` file before release |
| `flash-linear-attention`, `causal-conv1d` (optional; not required) | MIT, BSD-3-Clause | GitHub pages, checked 2026-10-20 |
| Unsloth (core Apache-2.0; some optional parts AGPL-3.0; `unsloth-zoo` LGPL-3.0 / AGPL-3.0) | checked 2026-10-20 | GitHub pages. **No `import unsloth` line was found by grep** in the 0.1 and 0.2 package files, the two launchers or the 8 trainer files; the 0.1 README also lists Unsloth under "Not needed". The author's training environment may still contain it, so the review must go through the **full package list**, not only the imports. |

Never in the repository: model weights, trained adapters, datasets, identity or system-prompt text, the Hugging Face cache, tokens or keys.

## 8. Before release

The author's rule: nothing is released until all testing for the current model is complete. Then, for this repository:

- [ ] Results ledger: **started** (one earlier run and the export record, in `benchmarks/RESULTS.md`). Still open: the summary and per-step log of the later run that produced the GGUFs, accuracy against bf16, and a log that records the package version.
- [ ] Run the v0.3 adjustments and the aligner end to end, and record the draft-head acceptance (overlap, agree) before and after.
- [ ] Restructure the launcher: today it loads a checksummed archive; a normal package layout is easier to install and review.
- [ ] Scrub: hard-coded drive-letter and `/mnt/...` paths, project-private names in file names, comments and default folder names, and a fresh secret scan. Start a **fresh git history**.
- [ ] Licence review of the full dependency list; confirm licence, copyright holder and `NOTICE`.
- [ ] Name check: "Pelias" is also an unrelated open-source geocoder ([pelias/pelias](https://github.com/pelias/pelias), MIT licence, maintained by Geocode Earth; **verified 2026-10-20**). The author chose the repository name `pelias-trainer`; a trademark check is still to do.
- [ ] A fresh clone, on a clean machine, installs and passes the CPU tests using only this README.

## 9. Roadmap and open questions

* **Open question (hypothesis, untested):** does this training path give a useful gain on very low-bit (1–3 bit) models? The inference engine cannot recover information those formats have already lost, so the training side is where any benefit would have to come from. Plan: a small side-by-side run once the 4-bit results are logged.
* **Open question:** how much faster, and how accurate, is the v0.3 probe against the uncached one? Plan: time both on the same prompt and compare outputs token by token (the code already checks the second token).
* **Planned:** a measurement of how fast the expert stream runs during training against the PCIe and RAM floors measured for the engine, using the same tools.
* **Planned:** other MoE families, once the Qwen3.6 path has logged results.
* **Open direction (author's expectation; no design exists yet):** once pelias-engine is finished, a more distinctive way to train MoE models may be identified. Proposed first step: revisit the training path with the engine's measured floors and router traces (which experts are actually used) as inputs, and state any idea as a hypothesis with a test before building it.
* **Planned (author's decision, 2026-10-20): a public demo model.** After both this trainer and pelias-engine are finalised, train a model on **non-private data** to show the whole pipeline works, and publish **sample GGUF files in several quantisations made by this trainer**, so anyone can test the engine separately. Needs: a public dataset whose licence allows redistribution (**to be checked**), a results ledger from logs, and kernels in the engine for each quantisation offered (today only Q4_K and Q6_K exist there). Not started.

## 10. Acknowledgements and provenance

*Draft. The author confirms the wording, exact model names and versions before release.*

* **The author's own work.** The research, the design and the training experiments are the author's. The project began as a training problem (section 3.1) and led to the inference engine.
* **Third-party code and libraries.** The trainer imports PyTorch, `transformers`, PEFT, Triton, `safetensors` and NumPy, and its export step uses llama.cpp (MIT). Their licence notices will be reproduced in `THIRD_PARTY_NOTICES.md` (still to be written); the licences found so far are in section 7, and PyTorch's and NumPy's are **not yet checked**. Every source, with its role, is in [`CITATIONS.md`](CITATIONS.md).
* **Models.** The base is **Qwen3.6-35B-A3B** (Apache-2.0, checked 2026-10-20). The released base checkpoints were used only as starting points: the author's models, built for private use and not distributed, were trained and fine-tuned from this base, from the Gemma 4 family (26B A4B and 12B) and from Qwen3-Coder (exact names and versions **to be confirmed**; base-model licences **to be checked** before release). **No model weights, adapters or training data are part of this repository.**
* **AI assistance.** Developed with the help of AI tools: Claude (Sonnet, Opus and Fable models, Anthropic), Gemini Pro and Gemini 4 Argon (Google; exact versions **to be confirmed**), used for coding, review, analysis and documentation. The author directed the work and ran the hardware tests. Which tool wrote which part is **not recorded**, so no per-file claim is made.
