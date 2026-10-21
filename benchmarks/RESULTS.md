# Results ledger (trainer)

**Status: pre-release, private.** This page ties each number below to the log it came from, with a label. It covers **one training run** with the Pelias 0.2 launcher, the **4-bit and 5-bit GGUF files** made from it, and the tests those two files have had so far. It is not a speed comparison with other trainers, and it is not a quality claim.

| Label | Meaning |
|---|---|
| **measured** | Printed by a run or a checker; the log is in `raw/` (redacted copy). |
| **derived** | Arithmetic on measured numbers; the arithmetic is shown. |
| **recorded** | Taken from the project's written export record (made during the export), not from a program's output. |
| **not in repo yet** | Exists on the author's machine, or is described by a log, but the file is not in this repository. |
| **unmeasured** | Not run. Each has a path to a number at the end. |

Machine: Ryzen 7 5800X3D, RTX 4070 12 GB, Windows 11 with WSL2 (Ubuntu 26.04.1 LTS, 48 GB given to WSL). Software printed by the run summary: PyTorch 2.14.1+cu130, transformers 5.17.0, PEFT 0.21.2, Triton 3.8.0, NVIDIA driver 617.42, WSL kernel 6.18.54.1. The model is the author's private fine-tune of Qwen3.6-35B-A3B; its weights, adapter and training data are not published.

---

## 1. What is covered, and what is not

| Question | Answer |
|---|---|
| Which trainer produced the run? | The **Pelias 0.2 launcher** (the verifier names it). The log does **not** record whether the package inside was 0.2 or the 0.2.1 patch. No separate 0.2.1 log has been found. |
| Pelias 0.1? | No benchmarks exist for it (author-stated). |
| 5-bit and 4-bit GGUF? | Yes: `Q5_K_M` and `Q4_K_M`, section 3 and 4. |
| 3-bit GGUF? | **None.** No 3-bit file or log was found; the logs here cover 4-bit and 5-bit files only. |

---

## 2. The training run (measured)

Raw: `raw/training-run-0.2/results_pelias02.json` (summary) and `raw/training-run-0.2/verification_pelias02.json` (the checker's output). Run date 2026-10-19.

| Quantity | Value |
|---|---|
| Optimizer steps | 616, numbered 1 to 616 with no gaps; all values finite |
| Wall time | 202.7 min |
| Training loss, first steps to last steps | 2.006 to 0.194 |
| Evaluation loss over 12 evaluations | 0.649 first, 0.0722 last (also the best) |
| Largest gradient norm | 2.82 (clipped to 1.0 by the optimizer) |
| Throughput, median (p10 to p90) | 247.85 tok/s (88.4 to 598.1) |
| Time per micro-batch, median | 4.55 s |
| Share of step time waiting for expert transfers | 26.8% |
| Peak allocated VRAM | 8.02 GiB |
| Highest GPU memory seen by `nvidia-smi` | 9,705 MiB of 12,282 MiB |
| Peak process RAM | 20.0 GiB |
| Swap growth | 0.0 GiB |
| Backends used | grouped expert matmul, Triton dequantisation; both start-up self-tests reported ok |
| Adapter | LoRA rank 16, alpha 32, 310 modules (10 x 4 attention + 30 x 5 DeltaNet + 40 x 3 shared expert), 21,166,080 trainable parameters, float32, 84,751,528 bytes; **no LoRA on the routed experts** |
| Settings | 2 epochs, learning rate 1e-4, gradient accumulation 4, token budget 4,096, 2 resident layers, layer-0 experts at 8-bit, seed 3407 |
| Data (private; counts only) | 1,300 rows: 1,232 train and 68 evaluation; 987,440 assistant tokens; longest row 4,002 tokens against a 4,096 limit; none truncated |

**Derived check.** 616 steps x 4 micro-batches = 2,464 = 2 epochs x 1,232 rows, which fits one row per micro-batch.

**Read the losses carefully.** Two epochs over 1,232 rows, evaluated on 68 rows. The fall in loss shows the run converged mechanically; it says nothing about quality, and with a set this small it can also mean the model has memorised its data. The quality checks are in section 4.

### What the checker said

The checker's own verdict, unchanged: `VERDICT: NOT COMPLETE AND SOUND (2 FAIL, 0 warn, 24 pass)`.

* **22 of the 24 passes and all INFO lines** cover the run itself: the log is present and has a `[done]` line, no traceback or out-of-memory text, steps contiguous, loss fell, evaluation loss fell, VRAM and RAM and swap within limits, the adapter is complete with every module paired, no expert LoRA, and no row over the length limit.
* **The 2 FAILs and 2 further PASSes are four post-run reply checks** of the fine-tuned private model. Two of the four failed. The prompts and replies are withheld (they are about the private model), and the file shows one `WITHHELD` row in their place. So the verdict stays as printed: the run is mechanically sound by the other 22 checks, and the model failed half of its four reply checks.

---

## 3. The 4-bit and 5-bit GGUF files (recorded)

Raw: `raw/gguf-export/export_record.md`, dated 2026-10-19. It records what was produced and was written **before** any server had loaded the files.

| File (name replaced by `test-model`) | Bytes | Size | Bits per weight | Quantise time | sha256 |
|---|---|---|---|---|---|
| bf16 | 71,066,995,008 | 66.2 GiB, 753 tensors | 16 | n/a | not recorded |
| `Q5_K_M` | 25,347,532,800 | 23.6 GiB | 5.71 | 703.7 s | `ccc6720d5f19bcad6e21e4a1825c708c7e0441d3caa5b062aa6dbd96af2f46f0` |
| `Q4_K_M` | 21,713,463,296 | 20.2 GiB | 4.89 | 735.3 s | `326e54aae9cb313b7469c7c210857c03272ce080c7d1215556faeb1123f7d3fe` |

* **Derived:** the 4-bit file is 14.3% smaller than the 5-bit one (21,713,463,296 / 25,347,532,800 = 0.857), about 3.4 GiB.
* **Method (recorded):** merge the LoRA into the bf16 base shard by shard, `W' = W + (alpha/r) B A` with scale 2, 310 of 310 modules merged; convert with llama.cpp's `convert_hf_to_gguf.py` (commit `2ca15f540`); quantise both files from the one bf16 file with `llama-quantize`. No importance matrix was used for `Q4_K_M`; for `Q5_K_M` none was used either, inferred from file times, not from a log line. Community builds that do use one may score differently.
* **Environment named in the record:** GCC 15.2.0, CUDA toolkit 13.0.88 for the llama.cpp build, PyTorch 2.14.1+cu130, transformers 5.17.0.
* **The Q4_K_M mix** is not uniformly 4-bit (6-bit output layer, 6-bit feed-forward down projections in some layers). The record says it checked only the first nine tensors of the quantiser log against the llama.cpp source.
* The record lists as not verified at that time: the server load log, output quality, speed, and whether 4-bit and 5-bit differ on this machine. Section 4 is what has been run since.

---

## 4. Tests of the two files so far (measured, stock llama.cpp)

Both files were served with stock llama.cpp `b11399-2ca15f540`, KV cache `q8_0`, sampling temperature 1.0, top-p 0.95, top-k 20, seed 20261019, thinking off.

### 4.1 Decode speed, same setup for both (experts on the CPU, context 32,768)

The raw logs are in the `pelias-engine` repository (`benchmarks/raw/stock-llama-cpp-speed/`, runs `20261019_231714` and `20261020_020015`). Median tokens/s at short / 4k / 12k prompts:

| File | Short | 4k | 12k |
|---|---|---|---|
| `Q5_K_M` | 29.2 | 28.5 | 27.0 |
| `Q4_K_M` | 32.1 | 30.7 | 30.4 |
| Q4 over Q5 (derived) | 1.10x | 1.08x | 1.13x |

### 4.2 Capability checks (derived tallies; the full logs are not published)

The program that ran these keeps every reply in its log and the log is marked private, so only counts are given. Counts were made by tallying the program's own verdict for each reply (2 repeats; the 131k run has 1).

| Suite (what it asks) | `Q5_K_M`, ctx 32,768 | `Q4_K_M`, ctx 32,768 | `Q4_K_M`, ctx 131,072 |
|---|---|---|---|
| needle (find one code in a 4,000- or 16,000-token log, hidden at 10 / 50 / 90%) | 12 of 12 | 12 of 12 | 12 of 12 at 16k, 32k, 64k, 120k tokens |
| tools (call a tool, use its result, or correctly not call one) | 10 of 10 | 10 of 10 | not run |
| recall | 4 of 4 | 4 of 4 | not run |
| loop | 3 of 4 | 3 of 4 | not run |

* In **loop**, one reply per file failed a check that is specific to the private model (details withheld). The `Q5_K_M` file also had one reply shorter than asked (461 words against about 1,200); the program flags that as information and counts it as ok.
* With 4 to 12 replies per cell, one failure moves the rate a lot: the program's 95% interval for 12 of 12 is 75.7% to 100%, and for 3 of 4 is 30.1% to 95.4%. These tests cannot separate the two files.
* A separate set of tests on the private model's style and behaviour was run on both files; it is **not** published.

---

## 5. Not measured yet, and the path to each number

| Unknown | Path |
|---|---|
| Accuracy loss of 4-bit and 5-bit against bf16 (KL divergence, perplexity) | the bf16 file is kept for this; run the comparison on the same text and log it |
| Training speed against other trainers on the same 12 GB machine | run a named alternative on the same data and settings, log both |
| Whether the expert stream is near the PCIe and RAM floors during training | time the stream alone, compare with the engine's measured floors |
| Effect of the v0.3 adjustments (peak-VRAM guard, cached probe, layer-input offload) | run with each switch off and on, same data |
| MTP draft-head overlap and agreement before and after alignment | run the aligner end to end and log its metrics |
| A run with the 0.2.1 patch | find or repeat a run whose log records the package version |
| 3-bit files | quantise a 3-bit file from the bf16 file, then repeat sections 4.1 and 4.2 |
| The per-step log of this run | `run.metrics.jsonl` (237 KiB, 616 rows) and `status.txt` (75 KiB) are present on the author's machine per the checker, in the run folder; **not in repo yet** |

---

## 6. About the files in `raw/`

* Local paths, the launcher's file name and the model's file name were replaced (`<engine-home>`, `<hf-cache>`, `<launcher-0.2>`, `test-model`). One build-host identifier was removed from the kernel string. JSON files were re-written with indentation; no number was changed.
* In `verification_pelias02.json` the four reply-check rows are replaced by one `WITHHELD` row. The verdict line and counts are untouched.
* In `export_record.md` one word was removed from a sentence in the list of unverified items (the sentence is otherwise unchanged).
* The model files, adapter, training data and the private model's replies are not in this repository.
