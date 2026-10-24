# PELIAS TRAINER — C++/CUDA MASTER SCAFFOLD

**A ground-up design for the fastest, most faithful LoRA trainer for Qwen3.6-35B-A3B on one RTX 4070 (12 GB).**

| | |
|---|---|
| Document | `PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md` — scaffold v1.0 |
| Date | 2026-10-23; the Wave 0 hardware results and the 10.7 GiB ceiling were folded in on 2026-10-24; the universal-use design points and the ruling on decision 7 were recorded the same day |
| Status | **Design, plus the Wave 0 hardware measurements. No trainer code exists yet.** The hardware probes (P-H1 to P-H10, P-H12, P-H13, P-M3; P-M4 is built but has not run) exist as one standalone program and were run on the GPU on 2026-10-23/24, on WSL2 and on Windows-native; their results are in 11.12 and have been folded into the chapters that use them. Every other probe is still a specification that has not run. Every number carries a label (see 0.2). Every unverified claim has a named probe (Chapter 11) that turns it into a measurement. |
| Replaces | `pelias-trainer 0.2.1` (Python/PyTorch, the trainer that produced the current model) |
| Companion work | planned, **not yet written** (paused at the author's request until the author's new engine information is in): the engine cross-check (`PELIAS_ENGINE_CROSSCHECK_SPEED_TO_CEILING.md`), the engine master notes (`PELIAS_ENGINE_MASTER_NOTES.md`) and the proposed edits to the engine blueprint (16.7) |
| Copyright | Copyright (c) 2026 John Lipka and the Pelias contributors |

---

## 0. How to read this document

### 0.1 What it is, in one paragraph

A complete plan — the order of operations, the architecture, every kernel, the numerics, the memory budget, the export path, the verification ladder, and a catalogue of probes — for writing a trainer from nothing in C++20 and CUDA. The trainer reads the **same GGUF file the engine serves**, trains LoRA deltas on the parts of the model that carry behaviour, and writes a GGUF back. It is built so that what is trained is, as closely as the hardware allows, what is served. There is no Python in the product. Python (PyTorch/transformers) appears in exactly one place: as a *test oracle*, outside the product tree.

### 0.2 The labels — the honesty system

Every number and every claim in this document carries one of these labels. If a line has no label it is prose, not a claim.

| Label | Meaning |
|---|---|
| **MEASURED** | Measured on the author's machine and recorded in a ledger or raw run file in the two repos. |
| **READ-SRC** | Read directly from source or from the served model file's header this session (llama.cpp commit `2ca15f540`; the served `Q4_K_M` GGUF). Exact, but it describes what the code/file *is*, not how fast it runs. |
| **PUBLISHED** | From an external source, cited in Appendix B. Not re-measured here. |
| **DERIVED** | Arithmetic from labelled inputs. The arithmetic was run by script; inputs are listed next to the result. |
| **ESTIMATE** | Engineering judgement. Always paired with a probe that replaces it with a measurement. |
| **UNVERIFIED** | A claim from a proposal document, a reviewer, or memory that has not been checked. Not relied upon until its probe passes. |
| **DECISION** | A design choice made here, with its reason. Reversible unless stated. |

Units: **GiB = 2³⁰ bytes, MiB = 2²⁰ bytes, GB = 10⁹ bytes.** "tok/s" for training means *processed tokens per second* (every token that goes through the network, assistant or not), which is the convention of the v0.2 ledger (inferred: 202.7 min × 247.85 tok/s ≈ 3.0 M processed tokens vs 1.97 M assistant tokens over two epochs — probe P-T10 pins the definition).

### 0.3 The author's standing rules this design obeys

1. **Ground-up.** No part of the product is a wrapper around PyTorch, PEFT, Unsloth, TRL or Triton. Libraries allowed in the product: the CUDA toolkit (runtime, cuBLASLt for dense bf16 where it wins), libllama/`common` from the pinned llama.cpp for tokenizer and chat-template parity (MIT), and a small JSON parser.
2. **Never truncate.** A sample longer than the training window is a hard failure with a report, never silently cut.
3. **VRAM ceiling 10.7 GiB** (author ruling 2026-10-23; it was 10.9 until the probe P-H8 showed that the card gives only 10.83–10.85 GiB once a CUDA context exists). A governor enforces it in code (Chapter 15). What the number is measured against is an open decision (16.7, decision 7).
4. **Lead with how it can work and how to test it.** Where something is hard, the text says what would make it work and what script would show whether it does.
5. **Provenance in every module.** Each source file carries one header line: `from the spec: <name>` (written from this document), `adapted from <X>, licence <Y>`, or `original`. No AGPL/LGPL code (e.g. Unsloth) is copied.
6. **Privacy.** No dataset content, persona/identity text, private equations or credentials appear in this document or the repos. Where this document needs to refer to them it says "the author's private dataset".
7. **No "first of its kind" claims in the repos.** (Those words may appear in internal notes; the public documents describe what was built and measured, and let others judge novelty.)

### 0.4 Contents

| # | Chapter | What you get |
|---|---|---|
| 1 | Executive summary | The design on one page; the speed and accuracy story with honest uncertainty |
| 2 | Fact base | Hardware, the model's real tensor map, the served graph, v0.2 baseline, engine facts, prior art, proposal fold-in |
| 3 | Principles and non-goals | The rules the design follows; what it deliberately does not do; the universal-use design points (3.3) |
| 4 | **Precision plan** | Which bits live where: 4-bit, 6-bit, the isolated higher-precision core (budgeted; 8-bit only as a candidate), 16/32-bit; why; how it is exported |
| 5 | Architecture | Process model, module tree, memory plan with true sizes, the streamer, the layer-major scheduler |
| 6 | Kernel specifications | Every kernel family: math, layout, precision, tiles, tests |
| 7 | Training algorithm | The step, LoRA, loss, optimizer, determinism, MTP |
| 8 | Data pipeline | Tokenizer and template parity, packing, masks, audits |
| 9 | Export | Adapter GGUF, merged-core GGUF, byte-splice, engine hand-off |
| 10 | Verification ladder | Three oracles, ten rungs, gates |
| 11 | **Probe catalogue** | 100 probes: question, method, pass/fail, output, when; and (11.12) the hardware results so far |
| 12 | Order of operations | Stages T0–T14 with deliverables and exit gates |
| 13 | Performance model | Roofline, sensitivity table, targets |
| 14 | Risks and kill criteria | Including the 12 parked problem areas |
| 15 | Safety rails | Governor, canaries, atomic checkpoints, WSL specifics |
| 16 | **Engine co-design audit** | Requirement-by-requirement compliance matrix against the engine invariants, drift register, reconciliation of an outside research report, bridges B1–B5 (bank-aware loss, byte budget, accuracy gates, export alignment and hot-set profile, dual-branch numerics), the MEASURABLES register, the contract with the engine |
| A–E | Appendices | Tensor name map (checked against the served file), sources, the 13 reviewer traps, decision register, arithmetic ledger |

### 0.5 Which proposal documents were folded in (answer to "are the proposal documents also folded in?")

Yes. All nine files in the author's proposals folder were read and folded in. Chapter 2.9 has the full table of what was adopted, what was corrected and why. In short:

| File | Used for |
|---|---|
| `pelias-trainerV0.3.md` | The five v0.2 numerical flaws (A1–A5), layer-sync accumulation idea, 7-gate idea |
| `pelias-trainerV0.3-dropin.md` | Cross-check of the Python-era fixes (persona names inside it were **not** carried over) |
| `pelias-trainerV0.3-revier1.md`, `pelias-trainerV0.3-revier2.md` | The 13 reviewer bug claims — kept as a *checklist of traps* to test the new code against; they concern PyTorch/transformers 5.x internals and do not apply line-by-line to a native trainer |
| `pelias-trainerV0.3-c++CUDA-path.md` | The six blind spots and the ground-up feasibility argument |
| `pelias-trainerV0.3.1-c++CUDAplan.md` | Section structure, six kernel families, seven gates |
| `pelias-trainerV0.3.1-c++CUDAplan1-mod1-3.md` | `hal_arena`, `roofline_solver`, `gguf_safetensors_io` modules; four-registry checkpoint |
| `pelias-trainerV0.3.1-c++CUDAplan2.md` | Scaffold tree, header graph, byte layouts, 3-stream DMA, 8-phase order, exporter and gate runner |
| `pelias-trainerV0.3.1-c++CUDApl-potential.md` | A claims document — every claim in it is treated as UNVERIFIED and routed to a probe |

Also folded in: the older rundowns in the author's notes folder and its backups (handoff, ledger, PLAN_2, the Pelias blueprint, the Unsloth notes), the full pelias-engine ledger and blueprint, and the pelias-trainer v0.2 results.

An outside, AI-generated engine research plan (2026-10-23) was also reviewed. It is treated as a list of hypotheses, not a source: each claim was checked against the engine's measurements and, where possible, the primary document it cites (Chapter 16.4).

---

## 1. Executive summary

### 1.1 The goal

Fine-tune **Qwen3.6-35B-A3B** (40 layers; 30 DeltaNet + 10 full-attention; 256 routed experts, top-8; 3 B active parameters per token) on **one RTX 4070 with 12 GB**, with a ceiling of **10.7 GiB of VRAM**, in the least wall-clock time and with the least possible gap between *what was trained* and *what is served*.

The reference point is `pelias-trainer 0.2.1`: **median 247.85 tok/s, 202.7 min for 616 steps** (MEASURED, trainer ledger run 1), trained against a group-64 affine 4-bit copy of the experts and exported by merging into bf16 and re-quantizing the whole model.

### 1.2 The design in twelve lines

1. **One native process**, C++20 + CUDA 13.0 (`nvcc 13.0.88`), `sm_89` only. No Python in the product. No autograd engine: every operation has a hand-written forward and backward.
2. **GGUF in, GGUF out.** The trainer reads the exact file the engine serves and trains against those bytes. (D1)
3. **The bulk is frozen and byte-exact.** The 256×40(+1) routed experts stay in their served `Q4_K`/`Q6_K` bytes (18.65 GiB incl. the MTP block), are streamed from pinned host RAM, and are **spliced byte-for-byte** into the output — zero expert drift by construction, proven by hash. (D2)
4. **The trained part is a small "core"**: shared experts, DeltaNet projections, attention projections (1.41 B parameters, 820 MiB in the served mix). LoRA (rank 16 by default, 21.2 M parameters, fp32 masters) on top of the served base. (D3)
5. **The core is exported as an isolated higher-precision block, sized to the engine's byte budget** (`Q6_K` by class where the merge probe allows, `Q8_0` only where it fits; or the served type when trained through the exact quantizer, Mode Q). The 4-bit base cannot hold a small learned delta (it is erased at the same type), and an all-8-bit core does not fit beside the engine's 7.5 GiB bank under the 10.7 GiB ceiling (Chapter 16). An unmerged GGUF LoRA adapter is kept as the zero-drift reference. (D4, Chapter 4)
6. **Layer-major scheduling with K micro-batches per layer sweep.** Expert bytes per optimizer step are fixed at 2 × 18.16 GiB = **1.49 s at 26.1 GB/s** (DERIVED), independent of how many tokens the step holds, so PCIe is hidden once a step holds ≳2.2–3.0 k tokens. The trainer becomes compute-bound. (D5)
7. **Frozen weights never receive weight-gradients**, so backward costs ≈1× forward matmul, not 2×. Total ≈ 20.4 GFLOP per token at T = 8192 (DERIVED).
8. **Two forward numerics, chosen by measurement**: a bf16 reference path, and a **serve-faithful int8 path** (q8_1-style activations × integer dot, the same arithmetic the served CUDA kernels use). The int8 path is both ~4× the tensor-core rate on this GPU (DERIVED) *and* closer to what is served; it is adopted only if its KL to llama.cpp is not worse. (D6)
9. **Routing tape**: the top-8 expert ids per token per layer are stored in the forward sweep, so the backward recompute is deterministic; the same tape gives training-distribution expert-usage statistics for the engine's hot bank. The router is frozen by default. (D7)
10. **Loss**: fused chunked cross-entropy in fp32 (no full-vocabulary logits tensor), step-wide token normalization (fixes v0.2's short-sample over-weighting). (D8)
11. **Verification is built in, not bolted on**: three oracles (llama.cpp = served truth; an fp64 reference written for this project; a PyTorch fp32 autograd model used only in tests), a ten-rung ladder, and 100 probes that turn every uncertain number into a measured one.
12. **Fifteen stages (T0–T14)**, each with an exit gate. Nothing proceeds on a failed gate.

Beyond the twelve lines there is one **optional, gated** mode (Mode L, 7.12, D21): train the router (21.5 M fp32 parameters, frozen experts) with a locality term so that more of each token's top-8 lands in the engine's VRAM hot bank. It costs 0.2 % FLOPs and 0.32 GiB, ships only if it passes four pre-declared gates (quality ≤ +1 % CE, hit rate ≥ +0.03, breadth, measured engine speed), and is supported by one paper's stacking result and one paper's router-only fine-tune — and contradicted at small scale by the first paper's headline result. It is a hypothesis with a kill criterion, not a promise.

The engine's newest measurements were audited against this plan in Chapter 16. Of 22 requirements, 7 were already met, 4 needed correcting (chiefly the Q8_0 default that does not fit beside the 7.5 GiB bank, and the wording of the accuracy gate), 6 are new specification, 3 rest on premises the engine's data does not support (notably that the 10 full-attention layers need priority residency: they route like the other 30, and holding them fully resident would cost 4.6 of the 7.5 GiB and lower the overall hit rate from 0.827 to 0.711), and 2 wait on probes. The bank-aware training options (a zero-miss hinge, a clamped-forward variant and a distillation anchor) are specified in 16.5 with their own gates and kill criteria; export alignment, the hot-set profile and the packed-layout ownership are in 16.5 B4; every number still to be derived is in the MEASURABLES register, 16.6.

### 1.3 The picture

```
            ┌─────────────────────────── host (WSL2 / Windows) ───────────────────────────┐
 dataset ─► │ tokenizer + chat template (libllama/common)  ─►  packer (docs aligned to 64) │
 (private)  │                                                                              │
 served     │ GGUF mmap ─► [pinned expert store 18.65 GiB]   [boundary store N_step×40×2048]│
 GGUF  ───► │                    │ 3 DMA streams                     ▲ │                   │
            └────────────────────┼───────────────────────────────────┼─┼───────────────────┘
                                 ▼ H2D, 2 slots × ≤498 MiB           │ │ D2H / H2D
            ┌──────────────────────────────── GPU (≤ 10.7 GiB) ─────────────────────────────┐
            │ frozen on device: spine (served bytes) + lm_head Q6_K + routers F32 + embed    │
            │ trainable on device: LoRA A/B (bf16 compute / fp32 master) + Adam states       │
            │ per layer l, for each of K micro-batches:                                      │
            │    norm → [DeltaNet | attention] → norm → router → grouped expert GEMMs (4/6-bit)│
            │    + shared expert (LoRA) → residual                                           │
            │ forward sweep l = 0..39, then loss (chunked CE), then backward sweep l = 39..0 │
            │ (recompute-from-boundary + backward, using the routing tape)                   │
            └────────────────────────────────────────────────────────────────────────────────┘
                     │ end of step: AdamW on LoRA → checkpoint (4 registries)
                     ▼
   export:  A) GGUF LoRA adapter (zero drift reference)   B) merged-core GGUF = byte-splice
            of every unchanged tensor + budgeted core  →  llama-server / pelias-engine
```

### 1.4 What speed to expect (and how sure we are)

Inputs, all labelled in 13: peak tensor rates for the 4070 derived from published RTX 4090 figures by SM × clock scaling (BF16 FP32-accumulate **58.3 TFLOPS** stock, 65.4 at the author's overclock; INT8 **233 TOPS** stock, 261 at OC — DERIVED; P-H3 has since measured the overclocked rates: bf16 **65.2** and int8 **259.5** TOPS sustained, 11.12; the stock-clock repeat has not run); work per token **20.4 GFLOP** (DERIVED, T = 8192); efficiency ε = achieved/peak on the GEMMs (**unknown until probes K1–K3**); an overhead factor f for everything that is not a big GEMM (norms, recurrences, softmax, gathers, launch gaps).

| Scenario (T = 8192, stock clocks, f = 0.75) | ε = 0.4 | ε = 0.5 | ε = 0.6 |
|---|---|---|---|
| bf16 forward path | 858 tok/s | 1,072 | 1,287 |
| int8 forward path (ε₈ = ε − 0.1) | 1,318 | 1,675 | 2,031 |
| *v0.2, measured median* | *248* | | |

**Targets (DECISION, to be re-based after Wave 0 probes):** minimum viable **600 tok/s** (2.4× v0.2), target **1,200 tok/s** (4.8×), stretch **2,000 tok/s** (8×). For the same two-epoch run (≈3.0 M processed tokens, DERIVED from the ledger) that is **84 min / 42 min / 25 min** versus 202.7 min. The 3.2–4× figure in the proposal files is **UNVERIFIED** and is not used.

### 1.5 What accuracy to expect (and what we do *not* claim)

The accuracy argument is structural, and each part of it has a test:

| Source of error in v0.2 | What changes | How we know |
|---|---|---|
| Trained on group-64 affine 4-bit; served `Q4_K_M` (different grid) | Trained against the served bytes | P-S/P-T parity rungs; export KL (P-X2) |
| Whole model re-quantized after the merge (experts moved) | Experts byte-spliced, untouched | Hash equality of every non-core tensor (P-X1) |
| Merge of a small delta into a 4-bit grid | Core merged into 8-bit (or 6-bit); adapter kept as reference | P-Q3, P-X2, P-X3 |
| bf16 reductions in SwiGLU backward / router gradient | fp32 reductions and pointwise math | Kernel unit tests vs fp64 (rung 1) |
| Per-micro-batch loss normalization (short samples over-weighted ~87×, per proposal A3) | Step-wide token normalization | P-T5 |
| Frozen shared-expert gate and norms | Optional trainable (fp32), off by default until ablated | P-T6 |

**We do not claim a benchmark gain.** v0.2's train→serve loss was never measured, so the size of the improvement from removing the mismatch is unknown. The plan measures it (P-X2 on the new model; P-Q3 on the old adapter) before anyone says a number.

### 1.6 The five biggest risks (details in Chapter 14)

1. **Kernel efficiency ε is unmeasured** — the whole speed story rests on it. *Mitigation:* Wave 0 microbenchmarks (P-H3, P-K1–K3) before any architecture commitment; the bf16 dequant-to-scratch + tensor-core GEMM path is the safe fallback.
2. **DeltaNet backward** is the hardest kernel set (chunked WY form, reverse state recurrence). *Mitigation:* an fp64 sequential oracle whose backward is derived in 6.7; llama.cpp already contains an fp32 sequential forward kernel to compare against; materialized-attention-style fallback exists for the attention layers.
3. **No ECC on an overclocked card.** Silent bit-flips are possible. *Mitigation:* canary kernels, bitwise-deterministic re-runs, checksum of the LoRA state per step (15.3).
4. **The served model is not one numeric function**: GPU and CPU paths quantize activations differently, and the fused GDN kernel is a sequential fp32 recurrence. *Mitigation:* the KL gate threshold is set from the *measured* llama.cpp float-spread (P-Q4), not from a round number.
5. **The core's extra bytes compete with the engine's hot bank.** An all-Q8_0 core (+0.59 GiB) breaches the 10.7 GiB ceiling at the 7.5 GiB bank; all-Q6_K (+0.28 GiB) fits at neither prefill micro-batch (0.16 GiB free at the default, none at `-ub 2048`; it was 0.36 / ≈ 0.11 GiB at 10.9), so the byte-budget solver can raise only some classes (DERIVED, on the engine's whole-card basis; 16.3.1 gives the other reading, which leaves about 1 GiB, and decision 7 in 16.7 asks which one is meant). *Mitigation:* a byte-budget solver per class (D22), Mode Q for zero-byte merges (D23), engine regression gate (P-X5, P-E4).

### 1.7 What to run first

Wave 0 (Chapter 11.10) contains **47 probes** that need **no trainer code** — small standalone CUDA microbenchmarks and llama.cpp-based scripts. They settle: real tensor throughput and clocks under load, PCIe and pinned-memory limits on WSL2, the llama.cpp float-spread, whether Q8_0 or Q6_K survives the merge (using the existing v0.2 adapter), tokenizer and chat-template parity, the norm/V-head conventions, and the CPU/GPU activation-quantization effect. The remaining stages are ordered so that each one is testable against something already trusted.

---

## 2. Fact base

Everything the design stands on, in one place, each line labelled. If a later chapter uses a number, it comes from here or is derived from here.

### 2.1 Machine and toolchain

| Item | Value | Label |
|---|---|---|
| GPU | RTX 4070 12 GB (AD104, `sm_89`): 46 SMs, 5,888 CUDA cores, 184 tensor cores, 36 MiB L2, 504 GB/s, 200 W, boost 2,475 MHz | PUBLISHED (Tom's Hardware review) |
| GPU overclock | memory +1000 MHz; core ≈ 2,775–2,790 MHz at 965 mV; **no ECC** | author-stated; stability is a probe (P-H4, P-H10) |
| Shared memory | 100 KiB per SM; 99 KiB max per block on CC 8.9 | PUBLISHED (LSU GPU course table); probe P-H1 reads the real values |
| CPU / RAM | Ryzen 7 5800X3D (8c/16t, **AVX2 only**, 96 MB L3); 4×16 GB DDR4-3200 (51.2 GB/s peak) | author-stated / PUBLISHED |
| Host bandwidth | CPU alone 45.9 GB/s; GPU zero-copy read alone 26.1 GB/s; together 47.6 GB/s = 93 % of peak | MEASURED (engine ledger) |
| OS | Windows 11 Pro (build 26300) host + WSL2 (kernel 6.18.54.1; 48 GB cap, 48 GB swap on D:); the trainer is designed to build on both WSL2 and Windows-native; the hardware probes were built and run on both (11.12), the trainer itself has not | author-stated / DECISION; MEASURED (build number, from the Windows facts run) |
| Pinned memory | **32 GiB (= 34.4 GB) pinned and held on both WSL2 and Windows-native** (P-H7; the largest size tried, so the limit is **not found**). The engine's earlier 20,448 MiB was the largest size it tried, not a limit. `RLIMIT_MEMLOCK` = 64 MiB does not stop `cudaHostAlloc`; use `cudaHostAlloc` (`cudaHostRegister` gives a device pointer ≠ host pointer on WSL2) | MEASURED (P-H7, 2026-10-23/24; engine ledger for the pointer rule) |
| Driver / toolkit | driver 617.42 (CUDA 13.4 capable); WSL2: `nvcc 13.0.88`; Windows: `nvcc 13.4` and Visual Studio 18 Professional (used for the probe build); **avoid toolkit 13.2** | author-stated; MEASURED (Windows facts run); PUBLISHED (Unsloth warning) |
| llama.cpp | commit `2ca15f540` with carried patches `mm_ids_dedup`, `quantize_signed_id`, `cuda_hole_fix`, `pin_cold_to_cpu` | READ-SRC |
| VRAM ceiling | **10.7 GiB**, absolute (author ruling 2026-10-23; 10.9 from 2026-10-22 until P-H8 measured the free memory). The basis it is measured on is open: 16.3 X15 and 16.7 decision 7 | author-stated (DECISION) |
| Idle VRAM | 579 MiB after a fresh WSL restart; Windows `dwm` can hold ~3 GB after long uptime — gate on `nvidia-smi` before a run. 2026-10-23/24: NVML used before the CUDA context 914 MiB (WSL2 run) and 1,031 MiB (Windows run); with WSL off `nvidia-smi` showed 683–744 MiB used; the GPU-memory counter put `dwm` at 884 MiB | MEASURED (earlier sessions; P-H8; Windows facts run) |
| Tensor-core peaks (4070) | BF16/FP16 FP32-acc **58.3 TFLOPS** (65.4 at 2.775 GHz); INT8 **233.2 TOPS** (261.4 at OC); FP8 FP32-acc 116.6 | DERIVED from published RTX 4090 rows by SM × clock (46/128 × 2.475/2.52); **P-H3 has replaced the overclocked values with 65.2 / 259.5 (row below); the stock values are still DERIVED** |
| Ada FP16 with FP16 accumulate | twice the FP32-acc rate on 4090 (330 vs 165 TFLOPS); marketing "233 (466)" for the 4070 are *sparse FP16-acc* figures — not usable for training | PUBLISHED (aiwiki table, NVIDIA forum thread) |
| Free VRAM (P-H8) | Total 11.99 GiB. Once the CUDA context exists CUDA reports **10.83 GiB free on WSL2 and 10.85 on Windows-native** (NVML 10.95 / 10.85); the same 10.83 appears in 21 engine logs, so it behaves as a fixed driver budget and not as desktop load (inference). The context costs 154 / 138 MiB; Windows holds 914 / 1,031 MiB before the process starts | MEASURED (P-H8, 2026-10-23/24; engine logs) |
| Sustained tensor rates (P-H3) and soak (P-H4) | bf16 65.2, fp16 64.9, int8 259.5, fp8-e4m3 129.8 T(FL)OPS at an SM clock of 2,760–2,775 MHz, 99.8 % of the burst rate, the same on both platforms; 30-minute soak: 0 output mismatches, a flat clock, 43–56 °C, ≈ 107 W of the 200 W limit | MEASURED (11.12) |
| Memory and PCIe (P-H2, P-H5) | VRAM read 523–534 GB/s, write 448–454; host DRAM read best 43.7–44.8 GB/s (4 threads); PCIe 4.0 x16: pinned H2D and D2H 26.3 GB/s each at 256 MiB, pageable 10–12 GB/s | MEASURED (11.12) |
| Windows-side settings | Hardware GPU scheduling off; display-timeout (TDR) settings at the Windows default; power plan Balanced; Resizable BAR on (BAR1 total 16,384 MiB); fixed 32 GiB page file; 63.9 GiB RAM as 4 × DDR4-3200; no display-driver errors in the 24 h before the runs | MEASURED (Windows facts run, 2026-10-23); the effect of the power plan and of ReBAR is not tested |

### 2.2 The model, from the served file's header

Read from `test-model` (the file the engine serves), this session. **READ-SRC.**

| Key | Value |
|---|---|
| architecture | `qwen35moe`; `block_count` = **41** (40 trunk layers + 1 MTP block `blk.40`); `nextn_predict_layers` = 1 |
| embedding length | 2048; vocabulary 248,320 (tensor shape); context length 262,144 |
| layer pattern | `full_attention_interval` = 4 → full attention at layers 3, 7, 11, …, 39 (10 layers), DeltaNet at the other 30; `blk.40` (MTP) is a full-attention block |
| full attention | 16 query heads, 2 KV heads, head dim 256 (key and value length), RoPE dimension 64, sections [11, 11, 10, 0], base 10,000,000 |
| DeltaNet ("ssm") | conv kernel 4; `group_count` 16 (q/k heads); `time_step_rank` 32 (v heads); state size 128; inner size 4096 |
| MoE | 256 experts, **8 used**, expert FFN 512, shared-expert FFN 512; no `expert_weights_scale` key in the header (so no extra scaling) |
| RMS epsilon | 9.99999974752427e-07 (= 1e-6 as float32) |
| file | 753 tensors; 21,713,463,296 bytes (20.22 GiB); alignment 32; tensor data starts at byte 10,991,104; every tensor offset is 32-byte aligned |

### 2.3 The served model's real byte map

This is the single most useful table in the design: it tells the trainer what to read, what to freeze, and what each piece costs. **READ-SRC** (header of the served file), sizes in MiB.

**Quantization mix actually in the file** (`Q4_K_M` as produced by `llama-quantize` from the merged bf16 model, no imatrix):

| Tensor class (GGUF name) | GGUF shape (inner dim first) | Types in the file | Count | Total MiB |
|---|---|---|---|---|
| `token_embd.weight` | 2048 × 248,320 | Q4_K | 1 | 272.8 |
| `output.weight` (lm_head, untied) | 2048 × 248,320 | Q6_K | 1 | 397.9 |
| `ffn_gate_exps`, `ffn_up_exps` (separate, not fused) | 2048 × 512 × 256 | Q4_K | 41 + 41 | 5,904.0 + 5,904.0 |
| `ffn_down_exps` | 512 × 2048 × 256 | **Q6_K ×21**, Q4_K ×20 | 41 | 4,410.0 + 2,880.0 |
| `ffn_gate_inp` (router) | 2048 × 256 | F32 | 41 | 82.0 |
| `ffn_gate_inp_shexp` (shared-expert scalar gate vector) | 2048 | F32 | 41 | 0.3 |
| `ffn_gate_shexp`, `ffn_up_shexp` | 2048 × 512 | Q4_K | 41 + 41 | 23.1 + 23.1 |
| `ffn_down_shexp` | 512 × 2048 | Q6_K ×21, Q4_K ×20 | 41 | 17.2 + 11.2 |
| DeltaNet `attn_qkv` (joint q,k,v) | 2048 × 8192 | Q6_K ×14, Q4_K ×16 | 30 | 183.8 + 144.0 |
| DeltaNet `attn_gate` (the z projection) | 2048 × 4096 | Q4_K | 30 | 135.0 |
| DeltaNet `ssm_out` | 4096 × 2048 | Q4_K | 30 | 135.0 |
| DeltaNet `ssm_alpha`, `ssm_beta` | 2048 × 32 each | Q4_K | 30 + 30 | 1.1 + 1.1 |
| DeltaNet `ssm_conv1d` / `ssm_a` / `ssm_dt.bias` / `ssm_norm` | 4 × 8192 / 32 / 32 / 128 | F32 | 30 each | 3.8 / ~0 / ~0 / ~0 |
| Attention `attn_q` (**joint query + output-gate**) | 2048 × 8192 | Q4_K | 11 (10 + MTP) | 99.0 |
| Attention `attn_k` | 2048 × 512 | Q4_K | 11 | 6.2 |
| Attention `attn_v` | 2048 × 512 | Q6_K ×7, Q4_K ×4 | 11 | 5.7 + 2.2 |
| Attention `attn_output` | 4096 × 2048 | Q4_K | 11 | 49.5 |
| Attention `attn_q_norm`, `attn_k_norm` | 256 | F32 | 11 each | ~0 |
| `attn_norm`, `post_attention_norm`, `output_norm` | 2048 | F32 | 41, 41, 1 | 0.3 each |
| MTP: `nextn.eh_proj` | 4096 × 2048 | Q4_K | 1 | 4.5 |
| MTP: `nextn.enorm`, `hnorm`, `shared_head_norm` | 2048 | F32 | 1 each | ~0 |

**Where the 6-bit expert layers are.** Trunk layers with `Q6_K` down-projections (experts *and* shared expert): **0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39** (20 layers) + the MTP block 40. The other 20 trunk layers (5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33) use `Q4_K` down-projections.
DeltaNet `attn_qkv` is `Q6_K` at layers 0, 1, 2, 4, 10, 13, 16, 22, 25, 28, 34, 36, 37, 38 and `Q4_K` elsewhere. Attention `attn_v` is `Q6_K` at layers 3, 7, 19, 31, 35, 39 (+40).

**Derived sizes (DERIVED, script-checked against the table):**

| Quantity | Value |
|---|---|
| One routed-expert layer | **432 MiB** (down = Q4_K) or **498 MiB** (down = Q6_K); gate+up = 288 MiB |
| 40 trunk layers of routed experts | 20 × 432 + 20 × 498 = **18,600 MiB = 18.164 GiB** (matches the engine ledger) |
| plus MTP layer (498 MiB) | 19,098 MiB = **18.65 GiB** pinned in host RAM |
| One expert (3 matrices) | 3 × 2048 × 512 = 3,145,728 parameters; one layer = 805,306,368; 40 layers = 32,212,254,720 |
| Params active per token | experts 1.007 B + shared experts 125.8 M + DeltaNet projections 755.0 M (in) + 251.7 M (out) + attention 272.6 M + routers 21.0 M + lm_head 508.6 M = **2.94 B** |
| LoRA targets in v0.2 (310 modules) | 10×4 attention + 30×5 DeltaNet + 40×3 shared expert; rank 16 → **21,166,080 parameters** (arithmetic reproduces it exactly) |

**The "core" — the trainable spine, by class** (40 trunk layers; MTP adds 47.2 M params):

| Class | Params | Served mix MiB | If all Q8_0 | If all Q6_K | bf16 |
|---|---|---|---|---|---|
| Shared experts (gate, up, down) ×40 | 125.8 M | 72.7 | 127.5 | 98.4 | 240.0 |
| DeltaNet in-projections (qkv + z) ×30 | 755.0 M | 462.8 | 765.0 | 590.6 | 1,440.0 |
| DeltaNet alpha/beta ×30 | 3.9 M | 2.1 | 4.0 | 3.1 | 7.5 |
| DeltaNet out ×30 | 251.7 M | 135.0 | 255.0 | 196.9 | 480.0 |
| Full attention q+gate, k, v, o ×10 | 272.6 M | 147.8 | 276.2 | 213.3 | 520.0 |
| **Total core** | **1,409.0 M** | **820.4** | **1,427.7 (+607)** | **1,102.3 (+282)** | **2,687.5** |

Reading this table: the whole trainable core is ~0.8 GiB in the served mix. Making it 8-bit costs **+0.59 GiB** of VRAM when served; 6-bit costs +0.28 GiB; keeping it bf16 would cost +1.82 GiB over the served mix (1.23 GiB more than 8-bit) and is never needed.

### 2.4 The served forward graph (the function the trainer must reproduce)

**READ-SRC:** llama.cpp `src/models/qwen35moe.cpp`, `delta-net-base.cpp`, and the converter `conversion/qwen.py`, commit `2ca15f540`. Notation: `x ∈ ℝ^{T×2048}`; `RMS(x; w) = x / √(mean(x²) + ε) · w`, ε = 1e-6, applied per token.

**Layer `l` (0…39):**

```
a   = RMS(x; attn_norm_l)
m   = DeltaNet_l(a)          if (l+1) mod 4 ≠ 0
    = Attention_l(a)         otherwise
h   = x + m
u   = RMS(h; post_attention_norm_l)
y   = h + MoE_l(u)
```

**MoE:** `p = softmax(W_r u)` over all 256 experts (`W_r` is F32, 256×2048); take the top 8 of `p`; renormalize the 8 weights to sum to 1; `MoE(u) = Σ_{e∈top8} w_e · E_e(u) + σ(g_sh · u) · E_sh(u)` where `E(u) = W_down ( SiLU(W_gate u) ⊙ (W_up u) )`, and `g_sh` is the F32 vector `ffn_gate_inp_shexp` — a **single scalar gate per token** for the shared expert.

**Full attention (layers 3, 7, …, 39 and the MTP block):**
`[q|gate]_h = W_q a` — the 8192-row projection is **interleaved per head**: for head `h` the rows `[512h, 512h+256)` are the query and `[512h+256, 512h+512)` the output gate; `k = W_k a` (2 heads × 256), `v = W_v a`; `q_h ← RMS(q_h; q_norm)`, `k_h ← RMS(k_h; k_norm)` (per-head, 256-wide); **IMRoPE** on the first 64 dims of each head with sections [11, 11, 10, 0]; causal softmax attention with scale 1/√256 and GQA (8 query heads per KV head); `o_h ← o_h ⊙ σ(gate_h)`; `m = W_o o`.

**DeltaNet (layers other than 3, 7, …, 39):**

```
qkv  = W_qkv a                          # 8192 = q 2048 | k 2048 | v 4096
z    = W_z a                            # 4096  (GGUF name: attn_gate)
β    = σ(W_beta a)                      # 32, one per v-head
g    = ssm_a ⊙ softplus(W_alpha a + dt_bias)   # 32, log-decay ≤ 0; ssm_a = −exp(A_log) < 0
qkv' = SiLU( causal_depthwise_conv1d_k4(qkv) )  # per channel, 4 taps, zero history at sequence start
q,k,v = split(qkv');  q ← L2norm(q) / √128;  k ← L2norm(k)    # per head, eps = RMS eps
k-heads repeated (tiled) 16 → 32 to match v-heads: v-head j uses k-head (j mod 16)
for each v-head and token t (state S ∈ ℝ^{128×128}):
    S_t = e^{g_t} S_{t−1}  +  β_t ( v_t − e^{g_t} S_{t−1} k_t ) k_tᵀ
    o_t = S_t q_t
o'   = RMS_plain(o; ssm_norm[128]) ⊙ SiLU(z)          # gated RMS norm, per v-head
m    = W_out o'                                       # 4096 → 2048
```

(`S_t = e^{g}S + β(v − e^{g}S k)kᵀ` is the single-token form executed by `build_delta_net_autoregressive`: scale the state by `e^{g}`, compute `S k`, form the delta `β (v − S k)`, add the outer product with `k`. The batched path in llama.cpp is a chunked equivalent (chunk size 64, triangular solve) or the fused CUDA kernel below.)

**Head:** `logits = W_lm · RMS(x_40; output_norm)` (Q6_K, untied from the embedding).

**MTP block `blk.40`:** `eh_proj` maps `concat(enorm(embedding of the next token), hnorm(trunk hidden state))` (4096 → 2048) into one extra full-attention + MoE block, then `shared_head_norm` and the shared output head. Exact wiring and alignment is probe **P-S5**.

**Converter facts that the trainer must respect** (READ-SRC, `conversion/qwen.py`):

| Transform | Consequence for a GGUF-native trainer |
|---|---|
| Norm weights have **+1 folded in** (`w_gguf = 1 + w_hf`) for every norm except the DeltaNet gated-norm `ssm_norm`, which is plain | Trainable norms are trained in GGUF form; the HF-parity oracle must add/subtract 1. Numeric check: P-S2 |
| `A_log` stored as `ssm_a = −exp(A_log)`; `dt_bias` stored as `ssm_dt.bias` | If gates/decay parameters are ever trained, the chain rule goes through `−exp` |
| `conv1d` squeezed to `(4, 8192)` | |
| **V-head reorder**: HF groups v-heads by k-head; GGUF stores them in **tiled** order (v-head `j` ↔ k-head `j mod 16`) for `in_proj_qkv` V rows, `in_proj_z` rows, `in_proj_a/b` rows, `A_log`, `dt_bias`, the V channels of `conv1d`, and the columns of `out_proj` | A GGUF-native trainer uses tiled order everywhere — no permutation in the product. The PyTorch oracle needs the inverse permutation (test code only). Check: P-S3 |

### 2.5 What the served numerics actually are

A trainer that reproduces the *math* but not the *arithmetic* can still drift. These are the arithmetic facts, READ-SRC unless marked.

| # | Fact | Why it matters |
|---|---|---|
| N1 | Activations between graph nodes are float32; norms, router (F32 weights), gates, softmax, the DeltaNet recurrence run in float32 | The residual stream is fp32 in the served model. A bf16 residual in the trainer is a deviation (probe P-Q5) |
| N2 | CUDA matmul against a quantized weight first **quantizes the activation** to 8-bit blocks and runs an integer dot product. MMQ uses `block_q8_1_mmq` (128 values; Q4_K uses the "DS4" layout carrying a half2 scale + sum per 32 values, Q6_K the "D4" layout); small batches use the matrix-vector kernel with `q8_1` | The served model computes with **int8 activations**. The "serve-faithful" int8 forward in 4.5 replicates this |
| N3 | CPU matmul against Q4_K/Q6_K uses `vec_dot_type = Q8_K` (256-value activation blocks), with an optional 8×8 interleaved repack that also covers `MUL_MAT_ID` | The served model has **two** activation-quantization schemes depending on whether an expert runs on GPU or CPU — a possible contributor to the KL tail the engine measured. Probe P-S6 |
| N4 | The fused gated-delta-net CUDA op (`gated_delta_net.cu`) is on by default for both autoregressive and chunked paths (`fused_gdn_ar = fused_gdn_ch = true`, resolved by a probe at context creation). It is a **sequential per-token fp32 recurrence**: one warp per state column, state in registers, `S = e^g S + k ⊗ δ` | A ready, trusted fp32 reference for DeltaNet forward parity. Our chunked kernels must agree with it to fp32-level tolerance; the graph fallback (unfused chunk path) gives a second reference (P-S7) |
| N5 | Attention in the served path goes through the fused flash-attention kernels (`fattn-*`), with an f16 KV cache by default | Believed to compute QKᵀ and PV in f16 with f32 accumulation — UNVERIFIED; matters for choosing fp16 vs bf16 inside our attention kernel (P-Q7) |
| N6 | GGUF LoRA adapters are applied at run time via `build_lora_mm` with scale = alpha/rank, on `wqkv`, `wqkv_gate`, `wo`, `ssm_beta/alpha`, `ssm_out`, shared experts, shared gate, `output`, `eh_proj`, the router, and via `build_lora_mm_id` on experts | An adapter-GGUF export is a first-class, zero-drift way to serve a trained delta; rsLoRA α/√r maps to GGUF alpha = α·√r |
| N7 | `llama-quantize` supports `--tensor-type name=type`, `--tensor-type-file`, `--output-tensor-type`, `--token-embedding-type`, `--imatrix`, `--allow-requantize`, `--pure` | A per-class precision assignment can be produced with stock tooling for cross-checks of our own writer |
| N8 | `examples/eval-callback` exists in the pinned tree and graph nodes carry names (`l_out-N` etc.) | Per-layer hidden-state dumps from llama.cpp for parity (P-S1) |

### 2.6 The baseline being replaced: pelias-trainer 0.2.1

**What it is (READ-SRC: the `pelias_v0.2` source tree, read this session).** Python/PyTorch on HF `transformers`; bf16 spine on the GPU with PEFT LoRA on 310 modules; routed experts re-quantized to **affine group-64 4-bit** (layer 0 at 8-bit) and streamed per layer from pinned RAM; a custom `autograd.Function` for the experts (recomputes inside backward); per-layer activation checkpointing; 2 resident expert layers; AdamW β = (0.9, 0.999), weight decay 0, gradient clip 1.0, linear warm-up over 5 % of the steps then cosine to zero; LoRA rank 16, α 32 (scale 2); seed 3407; gradient accumulation 4; cross-entropy in 256-position chunks. Export: merge the adapter into the **original bf16 shards** → `convert_hf_to_gguf` (bf16) → `llama-quantize Q4_K_M` (no imatrix).

**What it measured (MEASURED, trainer `benchmarks/RESULTS.md`):**

| | Run 1 | Run 2 (second phase) |
|---|---|---|
| Steps / wall time | 616 / 202.7 min | 114 / 37.9 min |
| Median tok/s (p10 / p90) | **247.85** (88.4 / 598.1) | 56.45 |
| Token budget per micro-batch / LR | 4096 / 1e-4 | 8192 / 5e-5 |
| Seconds per micro-batch | 4.55 | — |
| Stream-wait share of time | 26.8 % | 29.5 % |
| Expert traffic per micro-batch | — | 30.38 GiB in 72 fetches of 432 MiB |
| Peak VRAM / host RAM | 8.02 GiB / 20.0 GiB | 9.36 GiB |
| Largest gradient norm | 2.82 (clipped to 1.0) | |

Data (counts only; content is private): 1,300 rows (1,232 train / 68 eval), 987,440 assistant tokens, 2 epochs.

**What it got wrong or never measured** — each is a design input:

| # | Finding | Label | Consequence here |
|---|---|---|---|
| F1 | Trained against an affine group-64 grid; served as `Q4_K_M` (a different grid). The LoRA learned to cancel *that* grid's error. Train→serve loss was **never measured** | READ-SRC (`quant.py`, `30_export.sh`) | Train on the served bytes (D1); measure train→serve KL on every run (P-X2) |
| F2 | The merged spine was re-quantized to `Q4_K_M`, so a small learned delta met a 4-bit rounding step | READ-SRC | Isolated higher-precision core inside the engine's byte budget, or merge-aware training (Chapter 4, 4.10) |
| F3 | Loss is `total / n_tok` **per micro-batch**, then `/ grad_accum`. A short reply gets more weight per token than a long one (ratio = length ratio; 3,500 vs 40 tokens ≈ 87×) | READ-SRC (`train.py:257–261`) + DERIVED | Step-wide token normalization (D8); test P-T5 |
| F4 | Peak-VRAM statistics are reset **inside** the micro-batch loop and the log line is written only every `grad_accum`-th micro-batch, so ledger "peak" and per-step stats describe the last micro-batch only | READ-SRC (`train.py:254, 266`) | The 8.02 / 9.36 GiB peaks are lower bounds on true step peaks; the new trainer logs step-wide aggregates |
| F5 | One host synchronization per layer per pass for the routing plan (`counts_t.tolist()`), i.e. ≈ 3 × 40 per micro-batch | READ-SRC (`experts.py:48`) + DERIVED count | Device-side tile scheduling, zero host syncs in the layer loop (6.2) |
| F6 | Each micro-batch streams the 36 non-resident layers twice: 72 fetches, 30.38 GiB; at accumulation 4 that is ≈ 121.5 GiB of PCIe traffic per optimizer step | MEASURED + DERIVED | Layer-major scheduling: 36.3 GiB per step regardless of tokens (D5) |
| F7 | SwiGLU/router backward in the expert function computes `(d_out * y).sum(-1)` in the compute dtype `cdt = hidden.dtype` (bf16). PyTorch reductions accumulate in fp32 internally, so the real loss is the bf16 rounding of each product and of the result (~0.4 % relative per element), not a bf16 *summation*. The proposal's "catastrophic cancellation" wording is overstated, the direction is right | READ-SRC (`experts.py:166, 192`); the fp32-accumulation behaviour is PUBLISHED PyTorch behaviour, not re-tested | fp32 pointwise math and reductions in our kernels; kernel tests compare against fp64 |
| F8 | The frozen shared-expert gate and RMSNorms cannot adapt | READ-SRC (`SPINE_TARGETS`) | Optional trainable fp32 gates/norms, ablated (P-T6) |
| F9 | Proposal claims about "8 expert GEMMs and 6 dequantizations per layer per micro-batch" | UNVERIFIED (structure plausible: forward, checkpoint recompute, and backward recompute each run the expert function) | The new design recomputes once (from the boundary) and never re-dequantizes to a global buffer |

Note: the line numbers quoted in the proposal files (e.g. "train.py 229–234", "experts.py line 46") are a few lines off from the `pelias_v0.2` tree read here (257–261, 48). The proposals were written against a slightly different revision; the *content* of the claims F3, F4, F5 is confirmed.

### 2.7 Engine facts the trainer depends on

All from the pelias-engine ledger and raw files (MEASURED unless marked). The trainer is the engine's *upstream*: whatever it writes, the engine serves.

| Fact | Value | Why the trainer cares |
|---|---|---|
| Hot-bank economics | 7.0 GiB bank; live hit rate 0.775–0.779 (1-token), 0.733 (2–31 tokens); simulated 0.806; layers 0–2 lowest (0.61–0.65), layer 20 highest (0.90–0.92); by domain: agent 0.790, code 0.753, persona 0.799, reasoning 0.674; domain hot-set overlaps (Jaccard) are low | Fine-tuning moves the routing distribution. The hot bank must be re-profiled on the *trained* model. The routing tape (D7) provides the statistics for free |
| Float spread | stock-vs-stock KL between llama.cpp batch configurations is 0.0174 / 0.0183 on the engine's test set — larger than the 0.005 strict limit; `tier8_nobank` 0.0017, `tier8` 0.0106; reruns of the same configuration are deterministic. (Column meaning to be re-verified; P-Q4 re-measures on the trainer's eval set) | Every "KL ≤ x" gate in this document is set relative to the measured float spread, never to a round number (author ruling H6) |
| Dequant parity | the engine's cold kernel dequantizes bit-identically to ggml on sampled blocks | P-Q1 extends this to every tensor of the file |
| The MMQ crash class | the 9–31-token MMQ crash was `ids ? ids[blockIdx.x] : blockIdx.x` mixing signed and unsigned 32-bit so a `-1` sentinel became 4,294,967,295 (fix `quantize_signed_id`) | Rule for our kernels: no signed/unsigned sentinel mixing for expert ids; tests with empty experts and holes (rung 1) |
| MTP speculation | draft 2/3 at a 7.0 GiB bank: 12k-token decode 51.5–52.9 tok/s vs 45.4 base | Training the MTP block alongside the trunk (D12) protects this gain; P-T8 measures acceptance before/after |
| Host bandwidth | CPU alone 45.9 GB/s, GPU zero-copy alone 26.1 GB/s, together 47.6 GB/s (93 % of DRAM peak) | PCIe streaming and CPU work (boundary copies, data pipeline) compete for the same DRAM; P-H6 |
| Pinned-memory rules on WSL2 | `cudaHostAlloc` pointers are identical host/device; `cudaHostRegister` are not; 20.4 GiB pinned works (the trainer's P-H7 later held 32 GiB on WSL2 and on Windows-native) | Allocation policy in `hal_arena` (5.3) |
| Memory ceiling | engine P1 reference cell: 7.0 GiB bank, ubatch 4096 → peak 10.7 GiB; the 7.5 GiB bank peaked at **10.40–10.44 GiB at the default micro-batch** (stamps 130854, 132648) and at **10.67–10.69 GiB at `-ub 2048`** (stamps 132907, 141137), the setting that gives the fast 1,680 tok/s prefill; the 8.0 GiB bank at 11.0–11.19 GiB and the 8.5 GiB bank at 11.4 GiB (> ceiling). These peaks are whole-card figures (NVML used, including the Windows desktop's share) | Headroom under the 10.7 GiB ceiling at the 7.5 GiB bank is **0.26–0.30 GiB at the default micro-batch and 0.01–0.03 GiB at `-ub 2048`; less a 0.1 GiB margin = ≈ 0.16 GiB and none (−0.09) for everything the core adds** (DERIVED on the whole-card basis; at the earlier 10.9 ceiling it was ≈ 0.36 and ≈ 0.11). Read instead as a limit on the process's own allocations, the same peaks would leave ≈ 1.2 and ≈ 0.9 GiB (16.3.1; decision 7). The engine's own non-bank footprint will differ from the tier's, so P-E4 re-measures it at the production micro-batch. An all-Q8_0 core (+0.59 GiB) fits at neither; all-Q6_K (+0.28 GiB) fits at neither on the whole-card reading; the loss of bank to a smaller bank is ≈ −0.03 hit per 0.59 GiB (DERIVED by interpolating the engine's static-bank curve 7.0 → 6.41 GiB: 0.805 → 0.778). P-E4 measures the real peak with real merged cores; Mode L (7.12) is the trainer-side lever that could buy hit rate back |
| Machine state | BAR1 16 GiB-used mystery on WSL2; THP set to `always`; HAGS off; Afterburner settings; file-read speeds | Recorded in the engine notes; P-H8 re-checks real free VRAM before runs and (2026-10-23/24) records BAR1 on both platforms: total 16,384 MiB, used 0 MiB inside WSL2 and 16,355 MiB on Windows-native (meaning not examined) |

### 2.8 Prior art and what differs

| Source | What it shows | Label | Relevance |
|---|---|---|---|
| Unsloth Qwen3.5 fine-tune guide | bf16 LoRA for the 35B-A3B needs ~74 GB; MoE QLoRA (4-bit) not recommended; router fine-tuning off by default; example r = 16, α = 16; keep ≥ 75 % reasoning rows | PUBLISHED | Why a different architecture is needed on 12 GB; supports router-frozen default. AGPL — no code is copied |
| KTransformers v0.6.4 SFT | LoRA with CPU-resident experts (AMX/AVX-512); ~600–700 tok/s on Qwen3-30B-A3B on their hardware; Qwen3.6 not mentioned | PUBLISHED | Closest relative; needs AVX-512/AMX which the 5800X3D lacks, so the CPU-compute route is closed here; the GPU-compute + streaming route is the one this design takes |
| FlashQLA (Alibaba) | fused GDN forward/backward kernels; state recomputed in backward; 2–3× forward and 2× backward on Hopper | PUBLISHED | Template for the DeltaNet backward structure (6.7); Hopper-specific kernels do not transfer to `sm_89` |
| Parallelizing the delta rule (arXiv 2406.06484); Gated DeltaNet (arXiv 2412.06464) | WY/UT chunk form; chunk size 64 or 128; hidden states recomputed in backward; L2-normalized q,k | PUBLISHED | The chunked algorithm (6.7) |
| Cacheable by Design? (arXiv 2608.18261) | Pre-registered negative result: training routers for locality from scratch (137 M / 340 M) did not meet ≥ 30 % fewer misses at ≤ +1 % perplexity; untrained cache-aware rerouting costs +0.8–2.0 % at ≈ 30 % and +4–6 % at 50 %; on a locality-trained model the same rerouting costs +0.9 % instead of +7.7 % | PUBLISHED (read this session) | The earlier inference-time ruling stands for an *untrained* router; the stacking result is the evidence behind Mode L (7.12) |
| ReMoE (arXiv 2605.27081) | Router-only fine-tuning of a pretrained MoE with frozen experts; reuse +26 % relative; benchmarks within noise; CE-only control lowers reuse | PUBLISHED (read this session) | Same shape as Mode L; supplies the trust-KL idea and the CE-only control |
| Cache-Aware Joint Router Adaptation (arXiv 2609.04895); Sticky Routing (arXiv 2607.08780) | Post-training with auxiliary routers (abstract read); sticky-routing paper unreadable this session | PUBLISHED (abstract) / UNVERIFIED | Listed so they are not forgotten; neither is relied on |
| ESFT (arXiv 2407.01906) | train only task-relevant experts; gates not trained; shared parameters show more forgetting | PUBLISHED | Supports "experts frozen, router frozen" as the default; informs a later per-expert LoRA option; Mode L (7.12) is the deliberate, gated exception for the router |
| Cut Cross-Entropy (arXiv 2411.09009) | chunked linear+CE without materializing logits; gradient filtering at ε = 2⁻¹²; bf16-accumulation caveat, Kahan fix | PUBLISHED | Loss kernel (6.10); filtering off by default |
| rsLoRA (arXiv 2312.03732) | scale α/√r | PUBLISHED | Option; GGUF alpha mapping α·√r |
| MARLIN (arXiv 2408.11743) | fused 4-bit dequant GEMM near-4× faster up to batch ~16–32 | PUBLISHED | Evidence that dequant-in-register is the right shape for the cold-expert regime |
| llama.cpp quantization study (arXiv 2601.14277) | Llama-3.1-8B: Q4_K_M +0.24 PPL, Q6_K +0.03, Q8_0 +0.01 | PUBLISHED | Supports Q8_0 as near-lossless; magnitude on this model is P-Q3 |
| FlashAttention-2 README | head-dim 256 backward works on consumer GPUs (no dropout) from v2.5.5 | PUBLISHED | Fused attention backward at head dim 256 is feasible on `sm_89`; tile sizes are ours to find (P-K7) |
| HF architecture blog | confirms 40 layers, 30 GDN + 10 attention, 256 experts, top-8, expert FF 512, vocabulary 248,320, load-balancing coefficient 0.001 | PUBLISHED | Cross-check of 2.2 |

**What differs, stated without superlatives.** We did not find a published trainer that combines (a) training directly on the quantized GGUF bytes that a separate serving engine uses, (b) full-model streaming of frozen experts through a 12 GB card with layer-major accumulation, and (c) an isolated trainable core merged at export at a precision chosen against the serving engine's byte budget. That is absence of evidence from a search, not a claim.

### 2.9 Fold-in of the nine proposal files

| Source | Adopted | Corrected, and why | Routed to a probe |
|---|---|---|---|
| `pelias-trainerV0.3.md` | Layer-sync accumulation (→ D5); step-wide token normalization (→ D8); fp32 pointwise/reductions (→ 6.x); trainable fp32 gates/norms as an option (→ D3); the 7-gate idea (→ rung list); MTP co-training idea (→ D12) | "A2 catastrophic cancellation" is partly overstated (F7); "87×" is correct as arithmetic (F3); 421.88 MiB per layer is wrong — **432/498 MiB** (READ-SRC); "Q4_K_M / Q5_K_M training scheme with simulated quantizer" replaced by exact ggml bytes; "0.00 % drift" only for bytes we splice, not for the merged core | P-X1 (hash equality), P-X2 (KL) |
| `pelias-trainerV0.3-dropin.md` | Cross-check of data-pipeline ideas (multi-block `<think>` scan, strict-format tiers) | PyTorch/PEFT code is not used. Persona/identity names inside the file were deliberately not carried over | P-D5 |
| `…-revier1.md`, `…-revier2.md` | The 13 bug claims are kept as a *trap checklist* (Appendix C) | They concern `transformers 5.17` internals (`_gradient_checkpointing_func`, `saved_tensors_hooks`, `_update_causal_mask`) — not applicable to a native trainer; the ones with a native analogue (zero-sync routing, MTP expert-count assert, weight decay 0 on norms/gates, pad-token choice) become tests | Rungs 1, 6; P-D1 |
| `…-c++CUDA-path.md` | The six blind spots (host syncs, bf16 reductions, quantization mismatch, metric blindness, gates/norms, outlier-stretched grids) → each addressed in 6/7; ground-up feasibility argument | "28 tok/s probe" speed is unsourced → dropped; "AVX-512/AMX/DDR5" assumptions → this machine has AVX2/DDR4 | P-H5, P-H6 |
| `…-c++CUDAplan.md` | Section skeleton; six kernel families (mapped in 6.1); seven gates (→ ladder) | "9.8 GiB ceiling" → ruled **10.9 GiB**, since lowered to **10.7 GiB** (author, 2026-10-23); "dense 32–70 B" and "16–24 k context" have no evidence → out of scope / probe-gated; "3.2–4×" unverified → replaced by the sensitivity table | P-T10 |
| `…-c++CUDAplan1-mod1-3.md` | `hal_arena`, `roofline_solver`, `gguf_safetensors_io` module concepts (→ 5.2); four-registry checkpoint (adapters / shared gates / norms / MTP) (→ 7.9) | Fused `gate_up` layout → the real file has **separate** gate and up (READ-SRC); vocabulary 151,936 → **248,320** | P-X1 |
| `…-c++CUDAplan2.md` | Scaffold tree and header-graph discipline; byte-layout tables; 3-stream DMA with priorities (→ 5.5); 8-phase order (→ Chapter 12, extended to 15 stages); exporter + gate runner | "Nibble layout" text for Q4_K is wrong (Q4_K packs 32-byte chunks: low nibbles of 32 bytes = sub-block 2i, high nibbles = 2i+1) — fixed in 6.3; "bake" (re-quantize the merged weights at the same type) is not zero-drift; KL < 0.002 gate must come from the measured float spread | P-Q1, P-Q4 |
| `…-c++CUDApl-potential.md` | Nothing taken as fact | Every claim treated as UNVERIFIED and registered in Chapter 14 | per claim |
| Older rundowns in the author's notes folder (handoff, ledger, PLAN_2, blueprint, Unsloth notes) | Never-truncate policy; max-len 8192; hard-fail audits; identity-in-weights; `reasoning_content` handling for the `<think>` split bug; avoid toolkit 13.2; spawn-loop guard (a single native process removes the class); canary determinism for the no-ECC card | — | P-D2, P-H10 |

---

## 3. Principles and non-goals

### 3.1 Principles

| # | Principle | What it forces |
|---|---|---|
| 1 | **Train what is served.** | The trainer's base weights are the served GGUF bytes. Its forward pass reproduces the served arithmetic as closely as is useful (4.5). |
| 2 | **Splice, don't re-derive.** What is frozen is never recomputed, re-quantized or re-encoded; it is copied byte for byte and proven identical by hash. | Frozen tensors are mapped read-only; the exporter has no code path that could write them. |
| 3 | **Spend bits where behaviour lives.** The 94 % of parameters that are routed experts carry knowledge; the ~4 % in the spine carry the behaviour we want to change. | Experts stay at 4/6-bit untouched; the spine gets LoRA and an 8-bit home. |
| 4 | **Reorganize the loop before optimizing the kernel.** The old design was bandwidth-bound because the loop order was wrong. | Layer-major scheduling (5.6) comes first; kernel work comes second. |
| 5 | **Two implementations of everything.** A slow, obviously-correct version and a fast version; the fast one is admitted only when it matches. | Every kernel has a CPU or fp64 reference in `tests/` (rung 1). |
| 6 | **Measure, then plan.** The memory plan, the number of micro-batches and the slot count are functions of measured constants, not hard-coded. | `roofline_solver` reads `probe_constants.json` (5.2). |
| 7 | **No hidden state.** All device memory comes from arenas planned up front; all randomness is seeded; reductions have a fixed order in deterministic mode. | The step loop performs zero `cudaMalloc`; two runs with one seed are bitwise equal (P-T4). |
| 8 | **Fail loudly.** Over-length sample, NaN, ceiling breach, hash mismatch, canary mismatch: stop with a report. | No silent truncation, clamping, or retry loops. |
| 9 | **Reversible stages.** Each stage of Chapter 12 can be rolled back; no stage deletes its inputs. | Checkpoints are atomic; exports go to new directories. |
| 10 | **Python only in tests.** | Product tree has no Python; `tests/oracle/` may. |

### 3.2 Non-goals of this version

| Out of scope | Why |
|---|---|
| Dense models, 32–70 B | No evidence the design transfers; the proposals' claims for it are UNVERIFIED |
| Multi-GPU | One 12 GB card is the target |
| Full fine-tuning, expert fine-tuning | 74 GB class workload (Unsloth); experts stay frozen; a per-expert LoRA option is left as a clean extension point |
| Training the embedding table | Frozen; gather only |
| Context windows beyond 8,192 tokens per sample | The no-truncation policy is set at 8,192; longer needs a memory re-plan (parked, P-M2 shows headroom) |
| RL / DPO / preference losses | The loss interface (per-token weights, sequence masks) leaves room; not built |
| CPU-only or AVX-512/AMX training | The 5800X3D has AVX2 only |
| FP8 compute | Measured (P-H3, P-Q10) but not planned; Ada FP8 gives 2× BF16 FP32-acc rate, not 4× |
| Re-quantizing the base or computing an imatrix | The base is taken as given (D9); stock `llama-quantize` makes it |

### 3.3 Universal use — design points recorded 2026-10-24 (not built; they come after the top-tier gate)

The author's aim is a trainer that adapts to the machine, and to the MoE model the user chooses, instead of being tuned to one card and one file. The five points below are the author's, with the way the design would meet each. None of this is implemented, and nothing here adds a measured number: where a figure appears it is labelled where it is first used. Dense models stay a non-goal (3.2); "a different model" means a different MoE.

| # | Point (author, 2026-10-24) | How the design would meet it | Status |
|---|---|---|---|
| U1 | The system runs a quick probe at every start, takes the numbers it needs, and decides the route | A quick pass (free VRAM after the context, pinned PCIe rate, whether copies need the flush, a short tensor burst, free host RAM) writes a machine profile. The slow probes (30-minute soak, display timeout, pinned limit) run once per machine and are cached by GPU, driver, OS and clock state. The planner takes the lower of the quick and the cached value; the governor (15.1) stays the hard stop. The running time of the quick pass is unmeasured: it is the sum of its parts and is measured when it is built | DECISION (D29). The parts exist as P-H3, P-H5, P-H8, P-H12, P-M3 |
| U2 | The ceiling comes from the probe results, so the system knows the best number for the memory available (this is the ruling on decision 7) | Ceiling = the smaller of CUDA free and NVML free after the context exists, minus a margin. This is the process-only reading (P) of 16.3.1. 10.7 GiB stays this machine's value (10.83 free minus 0.13) until the margin is measured by the disturbance test (11.12) | DECISION (D28) |
| U3 | The probes leave the user a real choice of how to continue training | After the start-up probes a plan card lists two to four feasible plans. Each shows its fit (measured headroom), its predicted speed (DERIVED, or "unmeasured" with the probe that measures it), its quality risk (which of the gates K1–K3 covers it) and its effect on the export and the engine. One plan is marked as the default, the user chooses, and if only one plan fits the card says so. There are two layers. FIT: what the byte budget allows (core types, micro-batch, expert slots). WORTH: optional modes are offered only when their probes have passed (Mode Q after P-E6; Mode L and L-Z after P-L1, P-L2, P-E3; Mode B and N after P-E12). The gates do not choose a plan: they accept the plan that was trained or roll it back | DECISION (D29) |
| U4 | A different model is identified by its SHA-256 | Model key = the file's SHA-256 plus the ModelSpec fields read from its metadata. Today the base-file SHA-256 is only provenance (checkpoint manifest, export report, layout manifest, engine pack). Under U4 the per-model facts (hit rate, hot sets, expert bytes, stream time) are stored under the model key, apart from the per-machine facts. An unknown key means those facts are not reused and the model-specific probes (P-E1, P-E4, P-E5) run first. The hash is computed once and cached by path, size and modified time. The same model in another quantisation is a different key; whether routing facts carry over is a check probe, not an assumption | DECISION (D29). Hash time UNVERIFIED: it depends on the read rate (P-M4) and the CPU's SHA-256 speed, both to be measured |
| U5 | If a probe fails because of a memory spike, rerun it with a flush between runs; if that helps, training must flush too | The run scripts always write `vram_log.tsv`; `SETTLE=1` / `-Settle` waits for memory to return to the starting level before each probe; `RETRY_ON_FAIL=1` / `-RetryOnFail` reruns a failed short probe plain and then settled and prints a verdict (11.1, rule 8). If only the settled rerun passes, training adds a release-and-wait step at phase boundaries. If only the plain rerun passes, it needs a margin and a clean abort-and-resume. If neither passes, the limit is real | Built into the scripts on 2026-10-24 and tested against stubs only; no real failure has occurred yet |

---

## 4. Precision plan — which bits live where

> **The ask (author, 13:49):** move everything down to 4-bit, keep some 6-bit, have an 8-bit isolated core while training the rest of the active — "come up with the best path", with no hallucinated results.
>
> **The answer in one sentence:** the bulk stays exactly as served at 4/6-bit and is never touched; the part that learns lives in an isolated, higher-precision core (6-bit where the merge probe and the engine's byte budget allow, the served type with merge-aware training otherwise; 8-bit was the first suggestion and is a candidate only if the engine supports it and the budget has room) with fp32 learning state; the arithmetic that connects them is chosen by measurement, with a defined escalation ladder if a measurement fails.

### 4.1 The precision map

| Component | Params / size | At rest (served file) | Forward in trainer | Backward in trainer | Updated? | In the exported file |
|---|---|---|---|---|---|---|
| Routed experts: gate, up (all 41 layers) | 22.0 B · 11.5 GiB | **Q4_K** | dequant-in-register → bf16 MMA *or* int8 MMA | data-grad only, bf16, transposed dequant | **never** | **identical bytes** |
| Routed experts: down (**Q6_K in 21 layers** = 20 trunk + MTP; Q4_K in 20) | 11.0 B · 7.1 GiB | **Q6_K** / Q4_K | same | same | **never** | **identical bytes** |
| Token embedding | 508.6 M · 272.8 MiB | Q4_K | gather + dequant | none | never | identical |
| lm_head | 508.6 M · 397.9 MiB | **Q6_K** | dequant → logits fp32 (chunked) | data-grad only | never (option: LoRA, off) | identical |
| **Core**: shared experts, DeltaNet in/out/α/β, attention q/k/v/o | 1,409 M · 820 MiB | Q4_K/Q6_K mix | base bytes as above **+ LoRA** | data-grad through base + LoRA grads | **LoRA only** | **merged at the budgeted type (4.4)** |
| Router | 21 M · 82 MiB | F32 | fp32 | fp32 data-grad (weights of the selected 8 are differentiable, the selection is not) | frozen (Mode L, 7.12: fp32 trainable) | identical (Mode L: rewritten F32, exact) |
| Norms, conv1d taps, `ssm_a`, `dt_bias`, shared-expert gate | < 8 MiB | F32 | fp32 | fp32 | optional trainable (fp32) | F32 |
| LoRA A, B | 21.2 M | — | bf16 compute copy | bf16 → fp32 gradients | AdamW on fp32 masters | folded into the core / adapter GGUF |
| Adam moments | 21.2 M × 2 | — | — | — | fp32 | checkpoint only |
| Residual stream, layer outputs | per token 2048 | — | **fp32** inside a layer; boundary store fp32 (bf16 if RAM-limited) | fp32 | — | — |
| Softmax statistics, DeltaNet state, norms, router softmax, loss | — | — | fp32 | fp32 | — | — |
| Logits / cross-entropy | chunk × 248,320 | — | fp32 | fp32 | — | — |

(Sizes: gate + up = 2 × 41 × 144 MiB = 11,808 MiB; down = 4,410 + 2,880 MiB; all READ-SRC from 2.3.)

So, in the author's words: **everything bulky is at 4 bit, the 6-bit tensors the served mix already has stay 6-bit, the trained core is an isolated higher-precision block, and the learning state around it is 16/32-bit.** (8-bit was the first suggestion. The engine audit of 16.5 (B2) shows an all-8-bit core does not fit beside the engine's 7.5 GiB bank under the 10.7 GiB ceiling, so the default is the budgeted mix of 4.4; 8-bit stays a candidate for the classes that fit.) Nothing is *re-quantized to 4-bit* — the 4-bit is simply the file we were given.

### 4.2 Why train on the served bytes — three options considered

| Option | What it is | Verdict |
|---|---|---|
| O1 | Train on a simulated 4-bit grid (v0.2: affine group-64), serve a different one (`Q4_K_M`) | **Rejected.** The LoRA learns to cancel the wrong grid's error. Never measured, so unknown size; avoidable. |
| O2 | Train on bf16 weights (74 GB for this model; impossible on 12 GB), quantize afterwards | **Rejected.** Not feasible on this hardware, and post-hoc quantization of a merged delta has the same problem as v0.2's export. |
| O3 | Train on the exact served bytes; learn only a delta; merge the delta at higher precision | **Chosen (D1).** The base is identical to what is served, so the only train→serve gap is the core's merge. |

This is related to QLoRA-style training (a quantized frozen base plus a higher-precision learned delta). The difference is that the base here is the *final serving format* — `Q4_K`/`Q6_K` super-blocks as llama.cpp wrote them — rather than a training-only format such as NF4.

### 4.3 Why the core needs more than its served bits — the delta-versus-step problem, quantitatively

The trained delta for a matrix is `ΔW = (α/r)·B·A`. It must be written into the file. Whatever grid the file uses has a *step* — the smallest change the grid can represent. If ΔW is small relative to the step, writing it is lossy.

**Case 1 — merge at the same type (4-bit into 4-bit).** The served weights lie *exactly on* grid points. Adding a ΔW smaller than half a step and rounding back gives the same code: **the delta is erased** (not blurred — erased) for every weight whose |ΔW| < step/2, and replaced by a full step for those that cross. The result is a sparse, biased, noisy version of what was trained. (v0.2 escaped the *erasure* only because it merged into the original un-quantized bf16 shards, where the delta acts as a dither before rounding — a statistical merge that is unbiased on average but noise-dominated.)

**Case 2 — merge at 8-bit (Q8_0).** The 4-bit values (`d·sc·q − dmin·m`) are not on the 8-bit grid, but they round to it with error ≤ half an 8-bit step, which is small. The delta is larger than the 8-bit step for most weights, so it survives with bounded error.

A rough model (ESTIMATE; assumptions: Gaussian weights per 32-block, block range ≈ ±2.4 σ, uniform rounding): 

| Format | Step | Rounding noise (rms) | Typical ΔW of 3 % σ relative to step |
|---|---|---|---|
| Q4_K (15 levels) | 0.320 σ | **9.2 % σ** | 0.09 → erased |
| Q6_K (63 levels) | 0.076 σ | 2.2 % σ | 0.39 → mostly erased / dithered |
| Q8_0 (254 levels) | 0.019 σ | **0.55 % σ** | 1.6 → survives, ~5× amplitude SNR |

Noise ratio Q4_K : Q6_K : Q8_0 ≈ 17 : 4 : 1. The model already tolerates 9.2 % σ of noise per matrix (that *is* `Q4_K_M`; +0.24 PPL on Llama-3.1-8B per the published study, PUBLISHED), so adding 0.55 % σ is a variance increase of about 0.35 % of what is already there.

**Revision after the engine audit (Chapter 16, 2026-10-23).** The model above says 8-bit is the safest merge target. It does not say 8-bit is *affordable*: the engine's 7.5 GiB bank leaves 0.16 GiB (default prefill micro-batch) or none (`-ub 2048`) of VRAM under the 10.7 GiB ceiling for everything the core adds (it was ≈ 0.36 / 0.11 GiB at 10.9, and about 1 GiB if the ceiling is read as a limit on the process's own allocations: 16.3.1), depending on the prefill micro-batch (DERIVED from the engine's measured peaks, 16.5 B2), an all-Q8_0 core adds 0.59 GiB, and the engine's kernels are planned for `Q4_K`, `Q6_K` and `F32` only (the engine blueprint; Q8_0 is planned only for the KV cache). So the rule is now: **raise a class above its served type only as far as the byte budget and the engine's supported types allow, and use Mode Q (4.10) to close the remaining merge gap at zero extra bytes.** Q8_0 remains a candidate for classes where P-Q3 shows Q6_K is not enough *and* the budget has room.

**What this model does not tell us** (so it is a probe, not a fact): the actual size of ΔW on this model (3 % σ is an assumption; **UNVERIFIED**), whether Q6_K is enough for the larger-delta tensors, and the real functional effect. **P-Q3** measures ‖ΔW‖/‖W‖ per tensor class from the existing v0.2 adapter and the KL of each merge variant against the unmerged adapter, using only llama.cpp tools plus a small writer.

### 4.4 Export options for the core

| ID | Core in the exported file | Drift vs trained | Serving cost | Verdict |
|---|---|---|---|---|
| E-A | Base unchanged + **GGUF LoRA adapter** | none (the adapter is exactly what was trained, applied at run time by `build_lora_mm`) | ~620 extra small matmuls per token (310 modules × 2). If each costs 10–20 µs: +6–12 ms/token on a 24.6 ms/token engine timeline (ESTIMATE; P-X3 measures) | Always produced as the **reference**; not the production default |
| E-B | Merged into **Q8_0** | Q8_0 rounding noise (≈ 0.55 % σ, ESTIMATE) | +607 MiB VRAM vs served mix (+0.59 GiB) if applied to the whole core; same graph, no extra ops; a type the engine's weight kernels are not planned to support | **Candidate** per class, only inside the byte budget and if the engine supports the type (D22) |
| E-C | Merged into **Q6_K** for classes where P-Q3 shows ‖ΔW‖/step is high enough | Q6_K noise (≈ 2.2 % σ) | +282 MiB (+0.28 GiB) vs served mix if applied to the whole core; a type already in the served file and in the engine's kernel set | **Default candidate (D4 revised, D22)**, per class, if P-Q3 passes |
| E-D | Merged at the *served* type (Q4_K/Q6_K) | delta erased/dithered (Case 1) | 0 | **Rejected** |
| E-E | Merged to bf16/F16 | none | +1.82 GiB vs served mix | Rejected: nothing to gain over Q8_0 beyond rounding already below the float spread |
| E-F | **Mode Q (4.10):** the core is trained *through* the exact quantizer of its target type (served type or Q6_K), so the exported bytes are the bytes that were trained | none, by construction (the quantizer is bit-exact against ggml, P-E6) | 0 extra bytes at the served type; +282 MiB at most if all Q6_K | **Gated option (D23)**: tried when E-C fails P-Q3 for a class, or when the byte budget leaves no room |

"Isolated" means three concrete things: (1) the core tensors are their own named group in code (`TrainableCore` vs `FrozenStore`), so the type system prevents a frozen tensor from being written; (2) in the exported GGUF they are the only tensors whose data differs from the input file, which the exporter proves by hashing everything else; (3) at serving time they are the spine, resident on the GPU, so the engine can budget them as a block (the byte budget of 16.4, B2).

### 4.5 Forward arithmetic: two modes, chosen by measurement (D6)

**Mode R — reference (bf16).** Dequantize weights inside the kernel to bf16, use bf16 tensor-core MMA with fp32 accumulation; activations bf16 at MMA inputs, fp32 elsewhere. Simple, well understood, and the fallback.

**Mode F — serve-faithful (int8).** Quantize each activation row in blocks of 32 to int8 with an fp16 scale and block sum (the `q8_1` format that the served CUDA kernels use), take integer dot products against the weight codes with `mma.sync … s8` (weights as `u8`/`s8` codes), and apply the weight scales in fp32 per 32-sub-block — **the same arithmetic as llama.cpp's MMQ kernel**, except for fp32 summation order. With activation block scale `s_x` and integer codes `x̂`: for Q4_K each 32-wide sub-block `j` contributes `s_x·( d·sc_j·Σ(q·x̂) − dmin·m_j·Σ x̂ )`, using the stored block sum `Σ x̂`; for Q6_K each 16-wide sub-block contributes `s_x·d·sc_j·Σ((q−32)·x̂)`; for Q8_0 `s_x·d·Σ(q·x̂)`.

Mode F is attractive for two independent reasons: (a) **tensor-core int8 is 4× the BF16 FP32-acc rate on this GPU** (233 vs 58 T-ops, DERIVED), (b) it **is what is served**, so training through it (with a straight-through estimator for the activation quantizer) puts the LoRA in the same noise environment as inference — a QAT effect for free. Its risks: lower efficiency of the integer kernels than of bf16 ones (the scale-application epilogue), and accuracy of the straight-through gradient. The backward stays bf16 (4.7).

**How the default is chosen (not by argument):**

| Measurement | Rule |
|---|---|
| P-Q6: KL(trainer forward ‖ llama.cpp logits) on 64 held-out documents, mode R vs mode F | Prefer the mode with lower KL; both must be within the float spread (P-Q4) of llama.cpp |
| P-K1/K2: step time of each mode on layer shapes | Adopt F as default only if ≥ 1.3× faster **and** KL not worse |
| P-T3/P-T9: overfit and learning-curve parity | Both modes must train |

**Two activation schemes in the served model** (N2, N3): the GPU path uses `q8_1` blocks of 32; the CPU path uses `Q8_K` blocks of 256. Experts running on the engine's CPU see different activation rounding. Training in mode F matches the GPU path, which serves the hot bank (~78 % of expert activations at the measured hit rate). P-S6 measures how far the CPU path is from the GPU path in KL, so we know whether the remaining difference matters.

### 4.6 Residual stream, boundaries and internals

| Item | Plan | Why / test |
|---|---|---|
| Residual stream inside a layer | fp32 | The served model's residual is fp32 (N1) |
| Boundary store between layers (host RAM, K micro-batches × 40 layers) | fp32 if RAM allows, bf16 otherwise (2.50 → 5.00 GiB at 32 k tokens/step: 5.0 vs 10.0 GiB) | P-Q5: KL and loss-curve effect of bf16 vs fp32 boundaries |
| DeltaNet state and decay arithmetic | fp32; chunk-GEMM fragments bf16 or tf32 (decided by P-Q7 against the fp32 sequential llama.cpp kernel) | State error compounds over a chunk chain; this is the accuracy-critical recurrence |
| Attention | QKᵀ and PV on tensor cores in fp16 or bf16 with fp32 softmax statistics and accumulators | Served path is believed f16 (N5, UNVERIFIED); P-Q7 compares |
| Router | fp32 matmul and softmax; top-8 selection on fp32 | P-Q12 measures expert-set disagreement vs llama.cpp |
| Cross-entropy | fp32 logits per 256-position chunk; log-sum-exp in fp32; block-wise pairwise summation for the loss; gradient filtering (ε = 2⁻¹²) **off** | P-Q9 measures gradient cosine with filtering on; adopted only if ≥ 0.9999 and worth the time |

### 4.7 Backward arithmetic

* **Data-gradients through frozen quantized matrices** use the same dequant kernels read "transposed" (the reduction runs over the output dimension, not the quantized block dimension): dequantize a tile into shared memory, transpose with `ldmatrix.trans`, bf16 MMA, fp32 accumulate. Whether int8 can be used here too is probe **P-Q11** (cosine similarity of an int8 data-gradient against bf16); default is bf16.
* **No weight-gradient for frozen matrices.** Backward through a frozen matrix costs ≈ 1× its forward FLOPs, not 2×. This is the largest single reason the plan is faster than a full-bf16 estimate.
* **LoRA gradients**: `dB = (dY)ᵀ(XAᵀ)·scale`, `dA = (Bᵀ dY)ᵀ X·scale`, accumulated in fp32 with a fixed reduction order (deterministic mode) or atomics (fast mode, P-K13 measures the difference).
* **Optimizer**: AdamW on fp32 masters; bf16 compute copies refreshed after each step.

### 4.8 The escalation ladder — what to do if a measurement fails

Ordered cheapest-first. Each rung has the probe that decides whether to take it.

1. Boundary store bf16 → fp32 (P-Q5).
2. Forward mode F → R, or R → F (P-Q6).
3. Chunk-GEMM fragments bf16 → tf32 → fp32 in DeltaNet (P-Q7).
4. Attention internals fp16 ↔ bf16 ↔ fp32 (P-Q7).
5. Core class served type → Q6_K → Q8_0, each step only inside the byte budget and the engine's supported types (P-Q3, P-E4, P-E5).
6. Mode Q (4.10): train through the exact quantizer of the target type, closing the merge gap at zero extra bytes (P-E6, P-E7).
7. Core export → adapter-only (E-A) for a release that must be zero-drift, accepting the serving cost (P-X3).

### 4.9 What is deliberately *not* promised

* "4-bit trained beats Q5" and "0.00 % drift" (proposal claims) are **UNVERIFIED**. The first is untested; the second is true only for tensors we splice (hash-proven), not for the merged core.
* Nothing here claims a benchmark improvement. The precision plan's job is to ensure the gap between *trained* and *served* is small, measured, and attributable.

### 4.10 Mode Q — merge-aware training through the exact quantizer (D23; gated option)

**Idea.** If the file will hold `Q_T(deq(W_base) + ΔW)` for a target type `T`, train the model *as that function*: the forward pass uses the quantized-effective weights, so the exported bytes are exactly the weights the loss saw. The merge gap is then zero by construction, at any type — including the served type, which costs no extra bytes. This is quantization-aware training whose quantizer is the real one.

**Definition.** For each core tensor with target type `T ∈ {served type, Q6_K, Q8_0}`:

1. `W_eff = Q_T( deq(W_base) + s·B·A )`, recomputed once per optimizer step (weights do not change inside a step). `Q_T` is a bit-exact port of ggml's reference quantizer for `T` (`quantize_row_q4_K_ref`, `quantize_row_q6_K_ref`, `quantize_row_q8_0_ref`; MIT; provenance header "adapted from ggml-quants.c, licence MIT"), run on the device, with the CPU port as its oracle (P-E6).
2. Forward and recompute use `W_eff` through the ordinary G-family kernels (the bytes *are* a normal `T` tensor). There is **no separate LoRA branch** in the forward in Mode Q, which removes the LoRA epilogue work for those tensors.
3. Backward: the data-gradient through `W_eff` is the usual one. The gradient to `A`, `B` is the straight-through estimator `∂L/∂ΔW := ∂L/∂W_eff`, which gives exactly the ordinary LoRA formulas `dB = s·dYᵀ(X Aᵀ)`, `dA = s·(Bᵀ dYᵀ) X` evaluated at the quantized forward — **no dense weight-gradient is ever formed**.
4. `A`, `B` (fp32 masters) are the high-precision shadow that accumulates updates smaller than a grid step; weights flip when the shadow crosses a threshold.
5. At export, `W_eff` of the last step is written as the tensor — bit-for-bit what was trained.

**Costs (ESTIMATE; P-E7 measures).** Per optimizer step: read the core's base bytes (0.8 GiB), form the rank-16 delta in registers, quantize, write the packed result (≤ 1.1 GiB): memory traffic of ≈ 3 GB ≈ 10 ms at the card's bandwidth, plus the quantizer's arithmetic (the Q4_K and Q6_K reference searches iterate; the heaviest part). Against steps of seconds this is below 5 %. VRAM: base bytes (0.80 GiB, already in the plan) **plus** a `W_eff` copy (0.8–1.1 GiB) — inside the 3.0 GiB headroom of 5.3.

**Risks.** (1) STE through a coarse grid can stall or oscillate near thresholds; mitigations: learning-rate warm-up, an exponential moving average of `A`, `B` for the exported state. (2) Reproducing ggml's quantizer bit-for-bit on the GPU requires matching its floating-point operation order; the CPU port is exact by construction and is the fallback (slower, still affordable: ~1.4 B parameters per step on 8 cores). (3) In Mode Q the unmerged adapter GGUF is **not** the reference (training never saw `W_base + ΔW`); the merged file is the reference and the adapter is emitted for information only.

**Decision rule.** Try E-C (type raise, inside the byte budget) first because it needs no new machinery. Use Mode Q for any class where E-C fails P-Q3 or the budget has no room; run the ablation T-A11 (Mode Q at the served type vs LoRA-then-merge at Q6_K) and keep whichever has the lower held-out cross-entropy *after* export, measured through llama.cpp.

---

## 5. Architecture

### 5.1 Process model and coding basis

**One native process**, `pelias-train`, with sub-commands (`probe`, `audit-data`, `dump`, `train`, `eval`, `export`, `verify`). Everything the product does happens in this process; there is no Python interpreter, no `multiprocessing`, no RPC. (This removes the "spawn-loop OOM" failure class recorded in the older rundowns.)

**Coding basis (DECISION D13/D14):**

| Item | Choice | Reason |
|---|---|---|
| Language | C++20 host, CUDA C++ device (nvcc 13.0.88), PTX inline only for `mma.sync`, `ldmatrix`, `cp.async` | Tensor-core control; no dependency on CUTLASS/Triton |
| Target | `sm_89` SASS + `compute_89` PTX; tile constants in one header per kernel, derived from probe P-H1 (shared-memory opt-in, registers) | Ada only; PTX fallback keeps a newer driver working |
| Build | CMake ≥ 3.28, Ninja, `-lineinfo` always, `--ptxas-options=-v` logged in CI; one build tree for WSL2 (gcc/clang) and one for Windows-native (MSVC + nvcc) | Two supported platforms |
| Libraries in product | CUDA runtime/driver; cuBLASLt (optional, dense bf16 only); libllama + `common` (tokenizer, jinja chat template) from the pinned llama.cpp (MIT); a small JSON parser (nlohmann/json, MIT, already vendored by llama.cpp) | Token and template parity with the server for free; nothing else needed |
| Libraries in tests only | Python 3 + PyTorch + transformers + safetensors + `gguf-py`; llama.cpp tools (`llama-perplexity`, `llama-server`, `eval-callback`, `llama-quantize`, `llama-gguf-split`) | Oracles (Chapter 10) |
| Error handling | Status codes + RAII wrappers on the host; `CUDA_CHECK` on every call; no exceptions across the kernel-launch layer; device asserts only in debug builds | Predictable failure paths |
| Memory | Arenas planned before step 1 (5.3); the step loop contains **no** `cudaMalloc/Free`, no `new`; host pinned memory from one pool | P7 |
| Randomness | `philox` counter-based, seed + (step, layer, tensor) indexed; no global RNG state | Resumable and reproducible |
| Determinism | `deterministic` (default for gates and the first real run): fixed-order reductions, no float atomics; `fast`: allowed atomics | P-K13 prices the difference |
| Code layout rule | `kern/` depends only on `hal/` and plain structs of pointers and shapes. It never includes `model/`. `sched/` is the only place that composes kernels into layers | Keeps kernels unit-testable in isolation |
| Provenance | First line of every source file: `// provenance: from the spec: <name>` or `adapted from <X>, licence <Y>` or `original` | Author rule 5 |
| Licensing of references | llama.cpp MIT (ggml block structs and reference dequant: adapted, noted); FlashAttention-2 BSD-3 and flash-linear-attention MIT (algorithms studied, code not copied unless noted); KTransformers Apache-2.0; Unsloth: **nothing copied** | Clean-room discipline |
| Copyright header | `Copyright (c) 2026 John Lipka and the Pelias contributors` | Author rule |

### 5.2 Module tree

```
pelias-trainer/
├─ CMakeLists.txt  cmake/  third_party/   (pins: llama.cpp commit 2ca15f540, json)
├─ include/pelias/            public headers (stable surface for tests and tools)
├─ src/
│  ├─ hal/        device.{h,cc}  stream.{h,cc}  event.{h,cc}  arena.{h,cu}  pinned_pool.{h,cc}
│  │              vram_governor.{h,cc}  sync_audit.{h,cc}
│  ├─ plan/       probe_constants.{h,cc}  roofline_solver.{h,cc}  memory_plan.{h,cc}
│  ├─ io/         gguf_reader.{h,cc}  gguf_writer.{h,cc}  block_types.h  sha256.{h,cc}
│  │              safetensors.{h,cc}   (adapter / checkpoint interchange, test oracle exchange)
│  ├─ tok/        tokenizer.{h,cc}  chat_template.{h,cc}      (wrap libllama / common)
│  ├─ data/       dataset.{h,cc}  packer.{h,cc}  masks.{h,cc}  audit.{h,cc}
│  ├─ model/      spec.{h,cc}  frozen_store.{h,cc}  trainable_core.{h,cc}  lora.{h,cu}  tape.{h,cu}
│  ├─ stream/     expert_streamer.{h,cu}  boundary_store.{h,cu}
│  ├─ kern/       gemm_gguf/   dequant_ref.cu  mma_bf16.cu  mma_int8.cu  grouped_tiles.cu  dgrad_t.cu
│  │              route/       router_topk.cu  tile_map.cu  combine.cu
│  │              elementwise/ rmsnorm.cu  swiglu.cu  gated_norm.cu  residual.cu  cast.cu
│  │              attn/        attn_fwd.cu  attn_bwd.cu  attn_ref_blocked.cu  rope.cu
│  │              deltanet/    conv1d.cu  gates.cu  l2norm.cu  chunk_prep.cu  chunk_state.cu
│  │                           chunk_out.cu  chunk_bwd_*.cu  seq_ref.cu
│  │              loss/        ce_chunk.cu
│  │              optim/       adamw.cu  grad_norm.cu  lora_grad.cu
│  ├─ sched/      step.{h,cu}  layer_fwd.cu  layer_bwd.cu  head.cu  mtp.cu
│  ├─ train/      trainer.{h,cc}  schedule.{h,cc}  optimizer.{h,cc}  checkpoint.{h,cc}  metrics.{h,cc}
│  ├─ export/     export_adapter.{h,cc}  export_core.{h,cc}  splice.{h,cc}
│  ├─ verify/     gate_runner.{h,cc}  canary.{h,cu}  kl.{h,cc}  dump_compare.{h,cc}
│  └─ cli/        main.cc
├─ tests/
│  ├─ unit/       one executable per kernel family, fp64 CPU references
│  ├─ parity/     llama.cpp dump comparison harnesses
│  ├─ oracle/     Python (test-only): HF/PyTorch fp32 model, V-head permutation, tokenizer parity
│  └─ probes/     one directory per probe P-xx: program + README with pass/fail + expected outputs
└─ docs/          this document, model map, format notes, results ledger
```

**Header dependency graph (must stay acyclic; a CI script checks it):**

```
hal  ◄── plan, io, stream, kern, verify
io   ◄── model, export, data(masks need tok), verify
tok  ◄── data
kern ◄── sched
model ◄── sched, train, export
stream ◄── sched
sched ◄── train
data ◄── train
train, export, verify ◄── cli
```

**Module responsibilities (each is one reviewable unit; this is also the order they are written in Chapter 12):**

| Module | Responsibility | Key types / functions | Provenance |
|---|---|---|---|
| `hal` | Device selection, stream/event wrappers, arena allocator (bump + lifetimes), pinned pool, **VRAM governor** (hard 10.7 GiB, 5.3/15.1), `sync_audit` (debug mode: after every launch, record and check owner-stream rules) | `Arena::alloc(bytes, lifetime)`, `Governor::reserve()` | from the spec: plan1-mod1 `hal_arena`, rewritten |
| `plan` | Reads `probe_constants.json` (measured rates, bandwidths, smem), solves for window sizes, slot count, K, resident-tail length, and prints the plan before any allocation | `RooflineSolver::solve()` | from the spec: plan1-mod2 `roofline_solver` |
| `io` | Memory-maps the GGUF; indexes tensors; typed views of block data; writes a new GGUF by splicing; SHA-256 of tensor payloads | `GGUFFile`, `TensorView`, `splice_write()` | block structs adapted from ggml-common.h (MIT); rest original |
| `tok`, `data` | Tokenize and render exactly like the server; pack; mask; audit | `Tokenizer`, `render_chat()`, `pack_step()` | wraps libllama/common (MIT) |
| `model` | `ModelSpec` from GGUF metadata; `FrozenStore` (read-only mapped tensors: experts, embedding, lm_head, router, small tensors) and `TrainableCore` (LoRA state, optional gates/norms); routing `Tape` | | original |
| `stream` | Expert streamer (pinned store → device slots), boundary store (host copies of layer inputs) | `ExpertStreamer::acquire(layer)` | from the spec: plan2 3-stream DMA |
| `kern` | All device code (Chapter 6) | | mixed; each file notes its own |
| `sched` | The layer-major step: forward sweep, loss, backward sweep; enqueue-only (no host waits) | `run_step()` | original |
| `train` | Optimizer, LR schedule, checkpoint, metrics | | original |
| `export` | Adapter GGUF, merged-core GGUF, splice with hash proof | | original |
| `verify` | Gate runner, canaries, KL tools, dump comparison | | original |

### 5.3 Memory plan with true sizes

**Device (VRAM) — window T = 8,192 slots, step up to 32 k tokens.** Every row is either READ-SRC (from the file), DERIVED, or ESTIMATE (replaced by the number P-M2 measures).

| Item | GiB | Label |
|---|---|---|
| CUDA context + driver on WSL2 | 0.60 | ESTIMATE; MEASURED: the context itself costs 154 MiB (WSL2) / 138 MiB (Windows-native), and Windows already holds 0.89 / 1.01 GiB before the process starts (P-H8); P-M1 settles what the governor counts |
| Frozen core base (served bytes) | 0.80 | READ-SRC (820 MiB) |
| lm_head (Q6_K) | 0.39 | READ-SRC |
| Token embedding (Q4_K) | 0.27 | READ-SRC |
| Routers F32 (40) + small F32 tensors | 0.09 | READ-SRC |
| Expert slots: 2 × 498 MiB | 0.97 | READ-SRC (largest layer) |
| LoRA fp32 masters + grads + Adam m, v (r = 16) | 0.32 | DERIVED (21.17 M × 16 B) |
| LoRA bf16 compute copies | 0.04 | DERIVED |
| Resident layer-boundary tensors: current, next, gradient (fp32, 32 k tokens) | 0.75 | DERIVED (3 × 32,768 × 2048 × 4 B) |
| Per-layer working set for one window (forward buffers + recompute + backward) | 2.50 | ESTIMATE (accounting below) |
| Cross-entropy chunk (256 positions × 248,320, fp32 logits + workspace) | 0.65 | MEASURED in v0.2 (F152) |
| Routing tape and index buffers | 0.02 | DERIVED |
| *Mode L only (7.12): trainable router copy + gradient + Adam m, v (21.5 M × 16 B)* | *+0.32 (not in the total)* | *DERIVED; fits in the headroom* |
| Safety margin | 0.50 | DECISION |
| **Total** | **7.90** | |
| **Headroom under 10.7 GiB** | **2.8** | |

Headroom is spent, in order of value, by the roofline solver: a third expert slot (+0.49 GiB), more tape/window slots, a higher LoRA rank (r = 32: +0.32 GiB; r = 64: +0.96 GiB), keeping the last expert layers resident (each +0.42–0.49 GiB, saves one fetch per pass).

**Per-layer working-set accounting (T = 8,192; bf16 activations unless noted) — this is what P-M2 measures:**

| Sub-layer | Buffers alive at the peak | MiB |
|---|---|---|
| DeltaNet forward (kept for its backward) | `a` 32, `qkv` 128, conv-out 128, `z` 64, `β,g` 2, `w`,`u` 128, chunk-boundary states (128 chunks × 32 heads × 64 KiB, fp32) 256, `A` matrices (4096 × 16 KiB) 64, `o` 64, `o′` 64 | ≈ 930 |
| MoE forward (recompute) | `u` 32, router 8, gate/up outputs (65,536 assignments × 512 × 2) 128, `h` 64, per-assignment outputs 256, combined fp32 64 | ≈ 550 |
| MoE backward | per-assignment `dh,dg,du` 128, per-assignment `dx` 256, `dx` combined fp32 64, scratch ≈ 250 | ≈ 700 |
| DeltaNet backward (after MoE backward; forward buffers above still alive) | `dqkv` 128, `dz` 64, `do′` 64, running `dS` 64, scratch | ≈ 400 |
| Peak (at MoE backward: DeltaNet fwd + MoE fwd + MoE bwd) | | ≈ 2,200 → rounded to **2.5 GiB** |

A full-attention layer is smaller than a DeltaNet layer except for the attention score workspace (fused kernel: none; fallback blocked kernel: 2 heads × 256 MiB).

**Host (RAM; WSL2 cap 48 GB):**

| Item | GiB | Label |
|---|---|---|
| Pinned expert store (40 trunk layers + MTP, exact GGUF bytes) | 18.65 | DERIVED (19,098 MiB) |
| Pinned boundary store, fp32: 40 × N_step × 2048 × 4 B | 5.0 at 16 k tokens; 10.0 at 32 k | DERIVED (bf16 halves it) |
| Non-pinned: GGUF page cache for the non-expert tensors (~2 GiB), dataset tokens, checkpoints, runtime | ≈ 4 | ESTIMATE |
| **Total at 16 k / 32 k tokens per step** | **≈ 28 / 33** | |

Pinned total of 24–29 GiB exceeds the 20.4 GiB the engine had proven on this WSL2 setup → **P-H7** tested up to 32 GiB pinned and **held 32 GiB on WSL2 and on Windows-native** (11.12; the limit is above what was tried). Fallbacks, in order: bf16 boundary store; smaller step; boundary store in ordinary (pageable) memory with a pinned staging ring; experts as page cache with a pinned bounce buffer.

### 5.4 The streamer

**Host layout of the expert store.** One contiguous region per layer, `[gate_exps | up_exps | down_exps]`, each tensor exactly as in the GGUF, 4 KiB-aligned region starts. Per expert: gate = 512 rows × 8 blocks × 144 B = 589,824 B; up the same; down = 2048 rows × 2 blocks × 144 B (Q4_K) = 589,824 B or × 210 B (Q6_K) = 860,160 B. Layer total = 256 × (1,769,472 | 2,039,808) B = **432 | 498 MiB**. Expert `e` is at offset `e × (per-expert bytes)` inside each tensor. The store is filled once at start by reading the GGUF (≈ 20 GiB sequential read; the read path and the page-cache drop are probe P-M4).

**Slots.** `S` device slots of 498 MiB each (S ≥ 2). A slot's life: `FREE → LOADING(event E_load) → READY → IN_USE(event E_done) → FREE`. The compute stream waits on `E_load` before the first kernel that touches the slot; the streamer may reuse the slot only after `E_done` completed. The 40+1 layer indices are visited in a fixed order (forward 0…39, [MTP], backward 39…0), so the whole step's slot assignment is known before the step starts: **no data-dependent host decisions in the sweep.**

**Turnaround.** Layer 39 is the last forward layer and the first backward layer; the slot is kept, saving one fetch per step. With a third slot the MTP layer can be loaded while the head computes.

**Per-step expert traffic = 2 × 40 fetches − 1 = 79 fetches ≈ 35.8 GiB** (DERIVED: 2 × 18.164 − 0.486), independent of how many tokens the step contains (v0.2 moved ≈ 121.5 GiB per step at accumulation 4).

**Three streams (adopted from the proposals, D13).**

| Stream | Priority | Carries |
|---|---|---|
| `S_compute` | high | all kernels |
| `S_w` | normal | expert slot loads (H2D, 1 bulk copy per layer) |
| `S_x` | normal | boundary store reads/writes (H2D/D2H), tape spill if any |

Whether H2D and D2H genuinely overlap depends on the number of copy engines on this card and on the driver stack — **P-H12, measured (11.12):** the driver reports five copy engines, and on both WSL2 and Windows-native the three kinds of work **overlap only if the streamer flushes after enqueuing**. With ordinary submission the GPU timeline shows copies and kernels running one after another (all-three efficiency 0.47 / 0.52 for many short kernels, 0.00 for one long kernel); a `cudaStreamQuery(stream)` after each enqueue lets H2D, D2H and compute start together (all-three efficiency **0.92** on both; the H2D + D2H pair alone 0.84, each direction at about 87 % of its solo rate, DERIVED on WSL2). **Rule for the streamer (D27):** after enqueuing a slot load, a boundary copy or a kernel batch, call `cudaStreamQuery` on that stream; flush once per multi-millisecond batch (a 498 MiB slot load takes ≈ 20 ms), never after each microsecond-scale kernel. A flush costs host time only: 8.05 µs per call on a busy stream on Windows-native and 76.6 µs on WSL2 (P-H9), hidden while the GPU still has queued work. The solver assumes the measured efficiencies (0.84 for the pair, 0.92 for all three), not 1.0; the P-H12 gate is re-run on every driver change (R15), and if a driver loses the overlap the solver serializes `S_w` and `S_x` and adds the boundary traffic to the stream time in the roofline.

### 5.5 Boundary traffic (an honest accounting)

Layer-major order moves *activations* between host and device too. For a step of N tokens the per-layer traffic is 8 KiB × N per direction in fp32 (4 KiB bf16): forward writes one boundary snapshot per layer (D2H), backward reads it back (H2D) and keeps the gradient tensor resident.

| Step size N | Boundary traffic fp32 (2 × 40 × 8 KiB × N) | + experts 35.8 GiB | Time at 26.1 GB/s | Share of compute time at 1,500 tok/s |
|---|---|---|---|---|
| 4.9 k (≈ 4 documents, v0.2 parity) | 3.0 GiB | 38.8 GiB | 1.6 s | 49 % — hidden only if overlapped (fetch per layer 17–20 ms < compute per layer-pass ≈ 41 ms) |
| 16 k | 10.0 GiB | 45.8 GiB | 1.9 s | 17 % |
| 32 k | 20.0 GiB | 55.8 GiB | 2.3 s | 11 % |

(DERIVED at the engine's 26.1 GB/s; P-H5 measured pinned `cudaMemcpyAsync` bulk at 26.3 GB/s for H2D and for D2H at 256 MiB, so the times are about 1 % high. The time is still a lower bound: with copies in both directions at once the pair runs at efficiency 0.84 (P-H12, 5.4).)

**Consequence:** the design is stream-hidden for any step with ≳ 3 k tokens and comfortably so above 5 k. A step of one short document (e.g. 1.2 k tokens) would be stream-bound at ≈ 1.5 s regardless of speed — the packer therefore never forms steps below a minimum token count (default 3,072) except at the end of an epoch.

### 5.6 The layer-major scheduler

A *step* is: a packed set of documents of N tokens, split into `K = ⌈N / 8192⌉` windows of ≤ 8,192 slots (sizes from {2048, 4096, 6144, 8192} to avoid padding). For v0.2-like steps (≈ 4 documents, ≈ 5 k tokens) K = 1. For larger steps K > 1 and the same expert layer is used for all K windows before it is evicted.

```
STEP(tokens N, windows w_1..w_K, tape T):
  load tokens → embeddings (device gather+dequant of Q4_K rows) → x_cur[k]  (fp32, resident)
  ── forward sweep ──
  for l in 0..39:
      expert_streamer.acquire(l)                         # prefetch l+1 on S_w
      boundary_store.write(l, x_cur[·])      on S_x      # D2H snapshot of layer input
      for k in 1..K:
          layer_fwd(l, window k, x_cur[k] → x_next[k], tape T[l][k])   # records top-8 ids/weights
      swap(x_cur, x_next)
  ── head (per window, per 256-position chunk) ──
  for k: for chunk c:
      h = RMS(x_cur[k][c]; output_norm); logits = lm_head(h)  (Q6_K, fp32)
      loss_c, dlogits = softmax-CE(logits, labels, weights, 1/N_assistant_tokens_in_step)
      dh = lm_headᵀ dlogits  (data-grad only);  dx_cur[k][c] = RMS_bwd(dh, …)
  [MTP: blk.40 forward/backward with its own expert layer, D12]
  ── backward sweep ──
  for l in 39..0:
      expert_streamer.acquire(l)                         # slot for l+... reused at turnaround
      boundary_store.read(l) → x_in[·]       on S_x      # H2D, prefetched one layer ahead
      for k in 1..K:
          layer_bwd(l, window k, x_in[k], dx_cur[k] → dx_next[k], tape T[l][k]):
              recompute layer forward from x_in using the SAME top-8 ids (no router re-decision)
              backprop MoE → norm → DeltaNet/attention → norm;  accumulate LoRA grads (fp32)
      swap(dx_cur, dx_next)
  ── update ──
  grad_norm → clip(1.0) → AdamW on LoRA masters → refresh bf16 copies
  metrics (step-wide): loss, tokens, grad_norm, fetches, bytes, stream-wait, peak VRAM, tape expert-usage histogram
```

**Properties of this loop** that the tests check:

1. **No host synchronization inside the sweeps.** All work is enqueued; the host waits once at the end of the step (P-H9 decides whether the per-layer work is captured as CUDA graphs).
2. **The routing tape makes recompute deterministic**: backward uses the forward's top-8 ids and weights, never re-running top-k on slightly different numbers (a tie-break flip would silently change which experts get gradients).
3. **Equivalence:** a step of K windows equals the gradient of one big batch (P-T5); gradient accumulation is *inside* the layer loop, not outside it.
4. **Tape size** = 40 × N × 8 × (2 B id + 2 B weight as bf16 or 4 B fp32) ≈ 40 × 32,768 × 8 × 6 B = 60 MiB per 32 k tokens (host pinned, spillable) — negligible; and it is the expert-usage trace the engine's hot-bank profiler wants.
5. **What may differ between a single-window and a K-window step**: only fp32 summation order of LoRA gradients (and only in `fast` mode).

### 5.7 Document packing and masks (summary; details in Chapter 8)

Documents are laid into a window back-to-back, **each starting at a multiple of 64 slots** (the DeltaNet chunk size), the gap filled with a pad token that is masked out of loss, attention, and the DeltaNet state. Consequences: (1) DeltaNet chunks never straddle documents, so no segment-aware decay logic is needed in the hot kernels; (2) attention uses a block-diagonal-causal mask from `cu_seqlens`; (3) the conv1d resets its 3-token history at each document start; (4) padding waste ≈ 32 slots per document on average, i.e. ≈ 2.6 % for the ≈ 1.2 k-token average document length inferred from the v0.2 run (DERIVED: 3.01 M processed tokens / 2,464 sample passes; P-D4 measures the real figure). Pad slots still flow through the dense layers and MoE (wasted but harmless); a compaction pass is a later optimization (P-T13).

### 5.8 MTP block placement

The MTP block (`blk.40`) is one more full-attention + MoE layer with its own routed experts (498 MiB, in the pinned store) plus `eh_proj` and three norms. In the step it is run after the trunk forward (its input needs the trunk hidden states and the next-token embeddings) and its backward runs before the trunk backward sweep. Its LoRA modules (attention ×4, shared expert ×3, optionally `eh_proj`) form the fourth checkpoint registry. Exact wiring: probe **P-S5** (reads the llama.cpp MTP graph and compares with the proposals' "positions t, t+1 → t+2, masked across document boundaries").

---

## 6. Kernel specifications

### 6.0 Conventions

* **Layouts.** Activations are row-major `[T, F]` (token-major). Weights are exactly as in the GGUF: `[N_out rows][K_in]`, with the quantization blocks running along `K_in` (the reduction dimension of the forward GEMM). Windows are fixed `T ∈ {2048, 4096, 6144, 8192}` slots with `cu_seqlens` metadata in device memory (document starts, multiples of 64).
* **Dtypes.** `f32`, `bf16`, `f16`, `s8/u8`, `s32`. "fp32 accumulate" always. Statistics (RMS, softmax max/sum, log-sum-exp) always fp32.
* **Every kernel has**: (1) a header comment with its math, layout and provenance line; (2) a CPU/fp64 reference in `tests/unit/`; (3) a tolerance (6.11); (4) an entry in the *launch table* (`sched/launch_table.md`, generated) listing grid, block, shared memory, registers, spills from `ptxas -v`.
* **No kernel reads a host-written value that the host decided from device data in the same step.** All data-dependent shapes (expert counts, tile counts) live on the device; grids are sized to the worst case and surplus CTAs exit.
* **Index sentinels.** Expert ids and assignment indices are `int32` throughout; "absent" is `-1`; comparisons are done as signed. (This is the lesson of the engine's `ids ? ids[blockIdx.x] : blockIdx.x` signed/unsigned crash.) Rung 1 tests include empty experts, holes, and all-tokens-to-one-expert.

### 6.1 Kernel family map

| Family | Kernels | Replaces in v0.2 | Adapted from the proposals' "six families" |
|---|---|---|---|
| **G** quantized GEMM | G1 dense fwd (bf16), G2 dense fwd (int8), G3 grouped fwd + SwiGLU epilogue (bf16), G4 grouped fwd (int8), G5 dense dgrad, G6 grouped dgrad, G7 LoRA fwd/bwd, G8 dequant reference, G9 `q8_1` activation quantizer | Triton/torch dequant + grouped mm | family 1–2 |
| **R** routing | R1 router + softmax + top-8, R2 counting sort + tile map, R3 combine fwd/bwd (includes router-weight gradient) | `counts.tolist()` host sync, Python loops | family 3 |
| **E** elementwise | E1 RMSNorm fwd/bwd, E2 SwiGLU fwd/bwd, E3 gated RMSNorm (DeltaNet) fwd/bwd, E4 L2-norm fwd/bwd, E5 residual / cast / embedding gather | PyTorch eager | family 3 |
| **A** attention | A1 RoPE (IMRoPE), A2 fused fwd, A3 fused bwd, A4 blocked reference (also the fallback), A5 output-gate, A6 q/k per-head RMSNorm | SDPA | family 4 |
| **D** DeltaNet | D1 gates, D2 conv1d + SiLU (+L2norm epilogue), D3 chunk prep (triangular inverse), D4 chunk state sweep, D5 chunk output, D6 backward set, D7 sequential reference (fp64 and fp32) | HF reference loops | family 5 |
| **L** loss / optim | L1 chunked cross-entropy, O1 AdamW, O2 grad-norm + clip, O3 LoRA-gradient reduction | PyTorch ops | family 6 |

### 6.2 G — the quantized GEMM family

#### 6.2.1 Block formats (READ-SRC: `ggml-common.h`, `ggml-quants.c`)

| Format | Bytes / 256 values | Layout | Dequantized value |
|---|---|---|---|
| **Q4_K** | 144 | `d` (f16), `dmin` (f16), `scales[12]`, `qs[128]`. 8 sub-blocks of 32. `qs` is 4 chunks of 32 bytes; in chunk `c` the **low nibbles** are sub-block `2c`, the **high nibbles** sub-block `2c+1`. Scales/mins are 6-bit, packed: for `j<4`: `sc=scales[j]&63`, `m=scales[j+4]&63`; for `j≥4`: `sc=(scales[j+4]&0xF)\|((scales[j-4]>>6)<<4)`, `m=(scales[j+4]>>4)\|((scales[j]>>6)<<4)` | `d·sc_j·q − dmin·m_j`, computed as `(d·sc)·q − (dmin·m)` in fp32 |
| **Q6_K** | 210 | `ql[128]` (low 4 bits), `qh[64]` (high 2 bits), `scales[16]` (int8), `d` (f16). Per 128 values: for `l<32`: value `l` = `ql[l]` low nibble \| `qh[l]` bits 0–1; value `l+32` = `ql[l+32]` low nibble \| bits 2–3; value `l+64` = `ql[l]` high nibble \| bits 4–5; value `l+96` = `ql[l+32]` high nibble \| bits 6–7. Sub-block (16 wide) index of value `i` is `i/16` | `d·sc_{i/16}·(q − 32)` |
| **Q8_0** | 34 per 32 | `d` (f16), `qs[32]` (int8) | `d·q` |
| **q8_1** (activation) | 36 per 32 | `d` (f16), `s = d·Σq` (f16), `qs[32]` (int8); the MMQ variant groups 4 of them (128 values) | `d·q`; `s` is used for the Q4_K min term |
| Q8_K (CPU activation) | 292 per 256 | `d` (f32), `qs[256]`, `bsums[16]` | not used by the trainer (P-S6 only) |

These are the only formats the trainer must read (plus F32 and, for output, Q8_0). `block_types.h` carries the structs (adapted from ggml, MIT).

#### 6.2.2 G8 — dequant reference (the bit-parity anchor)

A trivially simple kernel (one thread per output element) that reproduces `dequantize_row_q4_K / q6_K / q8_0` **bit for bit in fp32** (same operation order: `d1 = d*sc; m1 = min*m; y = d1*q − m1`). It is the oracle for every other dequantizing kernel and for the exporter. Probe **P-Q1** runs it over *every tensor of the served file* against ggml's own functions.

#### 6.2.3 G1/G3 — forward GEMM with dequant-in-register (bf16 path)

`Y[M,N] = X[M,K] · Wᵀ`, `W` quantized along `K`.

* **Operand roles (swap-AB).** The weight tile is the MMA **A** operand (`m16 = 16 output features`), the activation tile the **B** operand (`n8 = 8 tokens`), so the result is `Yᵀ` fragments and dequantized weights go straight from registers into the A-fragment — no round trip through shared memory. (The same shape trick MARLIN and llama.cpp's MMQ use.)
* **Pipeline.** `cp.async` 3–4 stages: packed weight tile `[128 rows × 256 K]` (≈ 18 KiB for Q4_K) + activation tile `[128 tokens × 32..64 K]` bf16 into shared memory; weight fragments are read from shared memory as packed bytes and expanded: nibble extract (`lop3`/`prmt`), `cvt` to fp32, `fma(d·sc, q, −dmin·m)`, pack to `bf16x2`; activation fragments via `ldmatrix`. `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
* **Tiles.** CTA tile `(N_out 128) × (tokens 64 or 128) × (K 64)`; 4–8 warps; shared memory ≤ 96 KiB (≤ 99 KiB per block limit, P-H1). Tile shape is chosen per `M_e` bucket from the probe table (P-K1, P-K4). Tail tiles predicated.
* **Grouped (MoE) version G3.** The CTA's tile list comes from the device tile map (R2). A-side: activation rows are **gathered by index** (`sorted_token[r]`), so the 8× duplicated activation copy is never materialized. **Epilogue fusion:** gate and up are computed by the same CTA for the same output features (two weight tiles, two accumulators), then `h = silu(g)·u` is formed in fp32 and written once (bf16, or quantized to `q8_1` for the int8 down-projection). The down-projection G3′ scatters per-assignment results to `y_assign[pos, 2048]` (bf16) for the deterministic combine (R3).
* **Why dequant cost is affordable.** Per weight element the dequant is ≈ 4–6 ALU ops; each dequantized weight fragment feeds `tokens/8` MMAs (16 at a 128-token tile). Arithmetic intensity at `M_e = 256` is ≈ 900 FLOP per weight byte against a machine balance of ≈ 116 FLOP/B (58 TFLOPS / 504 GB/s, DERIVED) → compute-bound. Cold experts (`M_e < 32`) are memory-bound regardless; layer-major accumulation raises `M_e` (K windows) but the typical step has K = 1, where the load imbalance across experts is the practical limit (P-K4 uses real routing histograms from the engine traces).

#### 6.2.4 G2/G4 — forward GEMM, serve-faithful int8 path

Same structure, different arithmetic (this is the MMQ idea, implemented from the spec in 4.5):

1. **G9 quantizer** (one pass before each GEMM that consumes a new activation): per token row, per block of 32 along `K`: `d = amax/127`, `q = round(x/d)` (round-half-away, as ggml `roundf`), `s = d·Σq`; output `[T, K/32]` blocks. Bit-exactness against ggml's `quantize_q8_1` is probe **P-Q2**. For MoE, the `u` tensor is quantized once and shared by all eight experts' gate/up GEMMs; the SwiGLU epilogue quantizes `h` (512 wide = 16 blocks) for the down-projection.
2. **MMA.** `mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32` (A = weight codes as `u8` for Q4_K; B = `s8` activations); Q6_K uses `m16n8k16` with `s8` weight codes `(q−32)` because its scale granularity is 16; Q8_0 uses `s8 × s8` `m16n8k32`. (Integer MMA type combinations to be confirmed by compiling the probe P-K2 — PTX ISA, PUBLISHED, not yet compiled here.)
3. **Per-sub-block epilogue.** After each sub-block's integer MMA, convert the `s32` fragment and accumulate in fp32: `acc += float(iacc)·(d·sc_j·s_x) − (dmin·m_j)·(s_x·Σx̂)` (Q4_K). The row factors (`d·sc_j`, `dmin·m_j`) come from the weight side, the column factors (`s_x`, `s`) from the activation side.
4. **Cost model (DERIVED).** Per K = 32 step, the tensor core does 32 int8 MACs per output element (0.031 clk at 1,023 MAC/clk/SM); the epilogue does ≈ 3 fp32 ops per output element (0.023 clk at 128 lanes/clk/SM). They run on different units but compete for issue slots, so the realistic int8 efficiency is well below the bf16 one — which is why the sensitivity table assumes ε₈ = ε − 0.1 and probe P-K2 measures it.

#### 6.2.5 G5/G6 — data-gradient GEMMs ("transposed dequant")

`dX[M,K] = dY[M,N] · W` reduces over `N` (weight rows), not over the quantized block dimension. Plan: cooperatively dequantize a weight tile `[32..64 rows × 128 K]` into shared memory as bf16 (row-major `[n][k]`), then feed the MMA **B** operand with `ldmatrix.trans`; `dY` tile is the A operand. Same cp.async pipeline. Grouped version scatters `dX` per assignment (`dx_assign[pos, 2048]`) for a deterministic combine. Probe **P-K5** measures the efficiency of this path versus the forward kernel (expected lower: shared-memory round trip) and **P-Q11** tests an int8 variant.

#### 6.2.6 G7 — LoRA kernels

For a target matrix `W (N×K)` with `A (r×K)`, `B (N×r)`, scale `s = α/r`:

| Step | Math | Plan |
|---|---|---|
| Forward 1 | `t = X Aᵀ` ([T,K]·[K,r] → [T,16]) | skinny, memory-bound; reads `X` once (32 MiB at T = 8192, ≈ 65 µs at 504 GB/s) |
| Forward 2 | `Y += s·t·Bᵀ` | **fused into the base GEMM epilogue** (one `m16n8k16` MMA per output tile with the `B` tile `[128×16]` bf16 = 4 KiB in shared memory) — no extra pass over `Y` |
| Backward 1 | `u = dY·B` ([T,N]·[N,16] → [T,16]) | skinny; fused with the base dgrad when possible |
| Backward 2 | `dX += s·u·A` | epilogue of the dgrad GEMM |
| Gradients | `dB += s·dYᵀ t` ([N,T]·[T,16]) and `dA += s·uᵀ X` ([16,T]·[T,K]) | reductions over `T`; per-CTA partials in fp32 then fixed-order tree reduce (deterministic) or atomics (fast); accumulated across windows |

LoRA uses the **un-quantized** `X` (bf16/fp32), matching how the served graph applies adapters (`build_lora_mm` on the f32 activation). Probes P-K12 (efficiency of the fused epilogue) and P-K13 (deterministic vs atomic gradient reduction).

### 6.3 R — routing kernels

| Kernel | Does | Precision / notes |
|---|---|---|
| **R1 router** | `logits[T,256] = u·W_rᵀ` with `W_r` **F32**; fp32 FMA (no TF32 — top-8 selection must agree with the served graph, P-Q12); top-8 per token (warp-per-token iterative argmax, ties broken to the lower index — to be confirmed against ggml's `argsort` in P-S8); `w = softmax over the 8 selected logits` | The served graph does softmax over all 256 then renormalizes the selected 8; the 256-way denominator cancels, so the two are algebraically identical: `w_e = e^{l_e} / Σ_{sel} e^{l_{e'}}`. Computed as the 8-way softmax in fp32 (numerically better) |
| **R2 sort + tile map** | Counting sort of the `T×8` assignments by expert, stable in token order: block-local histograms → column scan across blocks → placement; then an exclusive scan of counts → expert offsets → tile list `(expert, row_start, rows)` for the chosen `M_tile`; all on the device | Deterministic (integer counts); upper-bound grid `⌈T·8/M_tile⌉ + 256` tiles; surplus CTAs exit |
| **R3 combine** | Forward: `out[t] = Σ_{j<8} w_{t,j}·y_assign[pos(t,j)]` in fixed `j` order, fp32 accumulate. Backward: `dy_assign[pos] = w·dy[t]` (generated on the fly in the A-loader of G6), `dw_{t,j} = ⟨dy[t], y_assign[pos(t,j)]⟩` (fp32), then the router backward below | Fixed order ⇒ deterministic; this replaces the proposals' `index_add_` |

**Router backward.** With `w = softmax(l_sel)`: `dl_e = w_e·(dw_e − Σ_{e'} w_{e'} dw_{e'})` for the 8 selected experts, `du += Σ_e dl_e · W_r[e,:]`. The selection itself is not differentiated; unselected experts get no gradient; `W_r` is frozen (no `dW_r`).

**Shared-expert gate.** `z = g·u`, `s = σ(z)`, `out_sh = s·E_sh(u)`: `ds = ⟨dy, E_sh(u)⟩`, `dz = ds·s·(1−s)`, `du += dz·g`, and if `g` is trainable `dg += Σ_t dz_t·u_t` (fp32, fixed-order reduction).

### 6.4 E — elementwise kernels (fp32 math, bf16/fp32 I/O)

| Kernel | Forward | Backward |
|---|---|---|
| **E1 RMSNorm** | `r = (mean(x²)+ε)^{-1/2}`, `y = x·r·w` (`w` already has +1 folded) | `dx_i = r·w_i·dy_i − (r³/n)·x_i·Σ_j dy_j w_j x_j`; `dw_j += Σ_t dy_j x_j r` (only if the norm is trainable) |
| **E2 SwiGLU** | `h = silu(g)·u`, `silu(g) = g·σ(g)` | `du = dh·silu(g)`; `dg = dh·u·σ(g)·(1 + g·(1−σ(g)))` |
| **E3 gated RMSNorm** (per v-head, width 128) | `n = RMS_plain(o; ssm_norm)`, `o′ = n ⊙ silu(z)` | `dz = do′ ⊙ n ⊙ silu′(z)`; `dn = do′ ⊙ silu(z)`; then the RMSNorm backward of E1 on `o` |
| **E4 L2-norm** (per head, ε = 1e-6) | `y = x/√(Σx² + ε)` | `dx = (dy − y·⟨dy,y⟩)/√(Σx²+ε)` |
| **E5** | residual add (fp32), `bf16↔fp32` casts, embedding gather + Q4_K dequant of the selected rows | transposes of the above |

### 6.5 Fusion rule

A fusion is admitted only if (1) the unfused kernels already pass rung 1, (2) the fused kernel passes the same tests *and* matches the unfused one bitwise in deterministic mode where the arithmetic allows, and (3) it removes at least one global-memory round trip of a ≥ 16 MiB tensor. The planned fusions: RMSNorm → quantize (G9); SwiGLU → quantize; conv1d + SiLU + L2-norm; LoRA-B in the GEMM epilogue; residual add in the output GEMM epilogue.

### 6.6 A — attention kernels (10 trunk layers + the MTP block)

**Data flow.** `a → (G1 with LoRA) [q|gate] (8192), k (512), v (512)` → split `q`, `gate` per head → **A6** per-head RMSNorm of `q` and `k` (256 wide, +1-folded weights) → **A1** RoPE on the first 64 dims → fused attention → **A5** `o ⊙ σ(gate)` → `W_o` (G1, LoRA).

| Kernel | Plan |
|---|---|
| **A1 IMRoPE** | 64 rotary dims = 32 pairs; sections [11, 11, 10, 0] select which of the (t, h, w, extra) position streams drives each pair. For text-only input all streams carry the same position, so the rotation angle is `p·θ_i`, `θ_i = base^{−2i/64}`, base 10⁷; **positions restart at 0 at every document start** (each packed document is a separate sequence). The pairing convention (adjacent pairs vs half-split) and the stream assignment are **UNVERIFIED** → probe **P-S4** compares our rotation with llama.cpp's post-RoPE tensors (via `eval-callback`) for both conventions on a text prompt. Backward: rotation by −angle |
| **A2 fused forward** | FlashAttention-style (online softmax, fp32 statistics, saves the log-sum-exp `L`). GQA packing: the **8 query heads that share a KV head are processed as extra rows** of the same tile (`rows = tokens × 8`), so each K/V tile is loaded once for 8 heads. Candidate tile: 8 tokens × 8 heads = 64 Q rows × 256 dims (32 KiB) + K tile 32×256 (16 KiB) + V tile 32×256 (16 KiB), K/V double-buffered ≈ 96 KiB — right at the 99 KiB limit; alternatives with 4 tokens × 8 heads. Causal + document block-diagonal mask from `cu_seqlens`; pad slots masked. QKᵀ and PV on tensor cores (`m16n8k16`, bf16 or f16 inputs, fp32 accumulate) |
| **A3 fused backward** | FA2 structure: `D = rowsum(dO ⊙ O)`, `P` recomputed from `L`, `dS = P ⊙ (dP − D)`, `dV = Pᵀ dO`, `dK = dSᵀ Q·scale`, `dQ = dS K·scale`; for GQA `dK`, `dV` accumulate over the 8 heads of the group. Head-dim-256 backward is the hard case on `sm_89` (shared memory 99 KiB; `dK`,`dV` accumulators 2 × tile × 256 fp32 held in registers). FlashAttention-2 reports head-dim-256 backward working on consumer GPUs from v2.5.5 (PUBLISHED), so it is feasible; the tile shapes are ours to find (**P-K7**). `dQ` computed in a separate deterministic pass (recompute `P`) rather than with atomics |
| **A4 blocked reference / fallback** | Two heads at a time: `S = QKᵀ` (batched bf16 GEMM, fp32 output, `T×T` = 256 MiB per head) → mask + fp32 softmax → `O = PV`; backward the same in reverse. Needed anyway as the oracle for A2/A3. Cost if it had to be the production path (DERIVED): ≈ 0.5 GB of extra traffic per head per layer ≈ 1 ms/head at 504 GB/s → 16 heads × 10 layers ≈ 160 ms per 8192-slot window against ≈ 5.5 s per window at 1,500 tok/s ≈ 3 % — acceptable as a fallback |
| **A5 output gate** | `o′ = o ⊙ σ(gate)`; backward: `do = do′ ⊙ σ(gate)`, `dgate = do′ ⊙ o ⊙ σ(gate)(1−σ(gate))` |
| **A6 q/k norm** | E1 with width 256 per head; `k` has 2 heads, `q` 16 |

FLOPs (DERIVED): attention score + value FLOPs per token ≈ 10 layers × 16 heads × 4 × 256 × T/2 = 0.67 GFLOP at T = 8192 — about 10 % of forward FLOPs, so a 2× slower attention kernel costs ≈ 5 % end-to-end. The DeltaNet kernels matter more.

### 6.7 D — DeltaNet kernels (30 trunk layers) — the hardest set

#### The math to implement (row convention: `S ∈ ℝ^{128×128}`, rows = key dim, columns = value dim; per v-head; `q` already scaled by 1/√128; `q,k` L2-normalized)

**Sequential form** (the oracle, and the served fused CUDA kernel):
```
a_t = exp(g_t)                              g_t ≤ 0
r_t = v_t − a_t · (k_t S_{t−1})             (1×128)
S_t = a_t S_{t−1} + β_t · k_tᵀ r_t          (outer product)
o_t = q_t S_t
```
**Chunked form** (llama.cpp's unfused graph, READ-SRC `build_delta_net_chunking`; chunk size C = 64; sequences padded at the end with `g=0, β=0, q=k=v=0`, which leaves the state unchanged):
```
g_cs_i = Σ_{j≤i} g_j  (within the chunk);   D_ij = exp(g_cs_i − g_cs_j) for j ≤ i
N_ij   = β_i (k_i·k_j) D_ij                 for j < i       (strictly lower, 64×64)
T      = (I + N)^{-1}                       (lower-triangular solve)
u      = T · (β ⊙ v)                        (64×128)
k_cd   = T · (β ⊙ e^{g_cs} ⊙ k)             (64×128)
kq_ij  = (q_i·k_j) D_ij   for j ≤ i
kg_i   = k_i ⊙ e^{g_last − g_cs_i};   g_last = g_cs_{C−1}
─ per chunk, with incoming state S ─
v_new  = u − k_cd · S                        (64×128)
o      = (q ⊙ e^{g_cs}) · S  +  kq · v_new
S     ← e^{g_last} · S  +  kgᵀ · v_new
```
Then `o′ = RMS_plain(o; ssm_norm) ⊙ silu(z)` per head.

**Why chunk-aligned documents matter.** Each document starts at a multiple of 64 (5.7), so the state is simply reset to zero at a document's first chunk, the conv history is zero there, and no chunk mixes documents.

#### Kernel plan

| Kernel | Parallelism | Work | Precision | Notes |
|---|---|---|---|---|
| **D1 gates** | token × 32 heads | `β = σ(W_beta a)`, `g = ssm_a ⊙ softplus(W_alpha a + dt_bias)`; pad slots get `β=0, g=0` | fp32 | `ssm_a` is stored negative; `W_alpha`,`W_beta` are tiny (2048×32) — G1 with N = 32 |
| **D2 conv1d + SiLU (+ L2-norm epilogue)** | token × channel | per channel 4-tap causal filter (`ssm_conv1d`, `(4, 8192)` F32), history zero at each document start, SiLU; epilogue: split `q|k|v`, L2-normalize each q/k head (ε 1e-6), `q ← q/√128` | fp32 math, bf16 store | Tap order (which tap multiplies the current token) is verified numerically by P-S1 |
| **D3 chunk prep** | chunk × v-head (4,096 items at T = 8192) | `g_cs`; `N` (64×64 from `K Kᵀ`); `T = (I+N)^{-1}` by blocked forward substitution (16×16 diagonal blocks + block GEMMs, as in FLA); `u`, `k_cd`, `kq`, `q⊙e^{g_cs}`, `kg` | triangular solve fp32 SIMT; the 64×64×128 products on tensor cores (bf16/tf32 inputs, fp32 accumulate) — choice by P-Q7 | `|N_ij| < 1` because `k` is unit-norm and `β ∈ (0,1)`; all exponents `≤ 0`, so no overflow. Saves `T` (64×64 fp32 = 16 KiB per item, 64 MiB per window) for backward |
| **D4 state sweep + output** | v-head × value-column slice (32 heads × 4 slices = 128 CTAs ≈ 2.8 waves on 46 SMs); **sequential over the 128 chunks** | the three per-chunk updates above; writes `o` and the incoming state `S_c` of every chunk (256 MiB per window, fp32) | `S` fp32 in registers/shared memory; MMA inputs bf16/tf32 with fp32 accumulate (3×TF32 split as an escalation) | Latency-bound chain: ≈ 4 small GEMMs per chunk; ESTIMATE ≈ 5–10 µs/chunk ⇒ ≈ 0.6–1.3 ms per layer-window ⇒ ≈ 1–2 % of step time (P-K8) |
| **D6 backward** | see below | | fp32 | |
| **D7 reference** | head × token | the sequential form in fp64 (CPU) and fp32 (GPU, mirrors `gated_delta_net.cu`) | | The two oracles; **P-S7** compares fp32 sequential against llama.cpp's fused and unfused paths |

#### Backward — structure and the oracle

**Oracle (fp64, sequential), derived here.** With `G = dS_t(from the future) + q_tᵀ dO_t`, `a = e^{g_t}`, `r = r_t`:
```
dq_t   = dO_t S_tᵀ
dβ_t   = k_t G r_tᵀ                dr = β_t k_t G
dv_t   = dr
dk_t   = β_t (G rᵀ)ᵀ  −  a (dr S_{t−1}ᵀ)
da_t   = ⟨G, S_{t−1}⟩ − ⟨dr, k_t S_{t−1}⟩            dg_t = da_t · a
dS_{t−1} = a G − a k_tᵀ dr
```
Upstream of the recurrence: `dg → d(alpha_raw)` through `ssm_a · σ(alpha_raw + dt_bias)`; `dβ → d(beta_raw)` through `β(1−β)`; `dq,dk` through E4 (L2-norm) and the `1/√128` scale; the 16 k-heads receive the **sum** of the gradients of the two v-heads that share them; `dv,dq,dk` through the SiLU and the 4-tap conv (adjoint taps, zero history at document starts). Verified in T8 by finite differences in fp64.

**Chunked backward (design, derived and checked in T8).** Recompute nothing global: use the stored `S_c`, `T`, and chunk tensors.
1. **Reverse state sweep** (sequential over chunks, reverse order, same CTA decomposition as D4): given `dO_c` and the carried `dS_{c+1}`:
   `d v_new = kqᵀ dO_c + kg dS_{c+1}`; `d kq = tril(dO_c v_newᵀ)`; `d(q⊙e^{g_cs}) = dO_c S_cᵀ`; `dS_c = (q⊙e^{g_cs})ᵀ dO_c + e^{g_last} dS_{c+1} − k_cdᵀ d v_new`; `d k_cd = −d v_new S_cᵀ`; `d kg = v_new dS_{c+1}ᵀ`; `d g_last ⊃ e^{g_last} ⟨dS_{c+1}, S_c⟩`; `d u = d v_new`.
2. **Per-chunk parallel stage:** `dT = du (β⊙v)ᵀ + d k_cd (β⊙e^{g_cs}⊙k)ᵀ`; `d(I+N) = −Tᵀ dT Tᵀ` → `dN` (strictly lower); then `dN → dβ, dk, dg_cs` through `N_ij = β_i (k_i·k_j) D_ij`; `dkq → dq, dk, dg_cs` through `kq_ij`; contributions from `e^{g_cs}`, `e^{g_last−g_cs}`; `dg = reverse-cumsum(dg_cs)` within the chunk.
3. Epilogue kernels: E3/E4/conv/gates backward as above, then the G5 dgrads for `W_qkv`, `W_z`, `W_alpha`, `W_beta` (with LoRA gradients).

FlashQLA (Hopper) and FLA (Triton) implement this structure, and recompute states in backward; we store them (256 MiB per window) because the layer backward immediately follows its recompute. If P-M2 shows memory pressure the fallback is FLA's recompute of `S_c` from checkpoints every 8 chunks.

### 6.8 L / O — loss and optimizer

| Kernel | Spec |
|---|---|
| **L1 chunked CE** | Per chunk of ≤ 256 positions **with non-zero loss weight** (rows with weight 0 are not computed: ≈ 35 % of positions are non-assistant in v0.2's data, DERIVED from 1.97 M assistant vs ≈ 3.0 M processed): `h = RMS(x; output_norm)`; `logits = h·W_lmᵀ` (G1/G2 with `N = 248,320`, Q6_K, fp32 output, 256 × 248,320 × 4 B = 242 MiB); per row `m = max`, `lse = m + log Σ e^{l−m}` (fp32); `loss = w_t (lse − l_label)/N_norm`; `dlogits = w_t (softmax − onehot)/N_norm` in place; `dh = dlogits·W_lm` (G5, data-gradient only) → RMSNorm backward. Loss is accumulated with block-wise pairwise summation in fp32 (no running fp32 sum over 10⁵ terms). Gradient filtering off |
| **O1 AdamW** | One fused kernel over the flat fp32 buffers `[master, grad, m, v]` of all trainable tensors; β = (0.9, 0.999), ε = 1e-8, weight decay 0.0 (0.0 on norms, gates, biases always), bias-corrected; the step's LR is a kernel argument computed on the host from `(step, schedule)` — no device-dependent host value |
| **O2 grad norm + clip** | Fixed-order two-level reduction of Σ g²; clip coefficient `min(1, 1.0/(‖g‖+1e-6))` applied inside O1 (v0.2 used clip 1.0; largest observed norm 2.82) |
| **O3 LoRA-gradient reduction** | Per-CTA partials of `dA`, `dB` per window → fixed-order tree reduce into the fp32 grad buffer (deterministic) or atomics (fast) |

### 6.9 Launch overhead and CUDA graphs

Per layer-window the forward launches ≈ 25–60 kernels and the backward ≈ 60–120; a K = 1 step has ≈ 40 × 180 ≈ 7,200 launches. At the measured 2.2–4.7 µs per eager launch (P-H9, the floor for an empty kernel) that is ≈ 16–34 ms per step against ≈ 3.3 s of compute — about 0.5–1 %, so graphs are an optimization, not a requirement. Where used: per layer *type* (DeltaNet-layer, attention-layer; forward/backward; per window bucket ⇒ ≈ 16 graphs), with all per-layer pointers (slot, LoRA, norms) read from a **device-side layer-descriptor table** so one graph serves every layer of that type. P-H9 measured graph capture and replay on this machine: replay is exact over 100,000 replays and a parameter update works, at 0.71 µs per kernel against 2.2–4.7 µs eager (68–85 % saving) on both platforms, so graphs are adopted where the layer structure allows; the engine already observed that graphs-off cost +4.5 ms per decode token, a different regime.

### 6.10 Where the FLOPs go (DERIVED, T = 8192, per token)

| Part | Forward GFLOP | Share of forward |
|---|---|---|
| DeltaNet projections (in + out + α/β, 30 layers) | 2.02 | 30 % |
| Routed experts (8 × 3 × 2048×512, 40 layers) | 2.01 | 30 % |
| lm_head | 1.02 | 15 % |
| Attention scores and values (10 layers) | 0.67 | 10 % |
| Attention projections | 0.55 | 8 % |
| Shared experts | 0.25 | 4 % |
| DeltaNet chunk core (rough) | 0.18 | 3 % |
| Routers | 0.04 | 1 % |
| **Total forward** | **6.74** | 100 % |

Training per token = forward 6.74 + recompute 5.72 (everything except the lm_head) + data-gradients 6.91 (matmuls 4.87 + attention 1.68 + DeltaNet core 0.36) + the lm_head data-gradient 1.02 = **20.39 GFLOP** (no weight-gradients for frozen matrices; LoRA gradients are negligible). Of this, **10.68 GFLOP (52 %)** is int8-eligible in the serve-faithful forward path.

**What this says about priorities.** The dense spine GEMMs (DeltaNet and attention projections, shared experts) plus the lm_head are 57 % of the forward work and are the *easy* case for tensor cores (large M, regular shapes). The grouped expert GEMMs, the part everyone worries about, are 30 %. Kernel work is therefore ordered G1 (dense) → G3 (grouped) → D (DeltaNet) → A (attention) in Chapter 12.

### 6.11 Rung-1 tolerance targets (DECISION — initial; tightened after the first results)

| Kernel | Reference | Initial tolerance |
|---|---|---|
| G8 dequant | ggml `dequantize_row_*` | **bit-exact** |
| G9 `q8_1` quantizer | ggml `quantize_q8_1` | **bit-exact** (codes, `d`, `s`) |
| G1/G3 bf16 GEMM | fp64 GEMM of the *same bf16-rounded inputs* | rel. L2 ≤ 1e-6 (accumulation order only) |
| G2/G4 int8 GEMM | CPU integer reference (exact integer part) + fp32 epilogue | integer part bit-exact; output rel. L2 ≤ 1e-6 |
| G5/G6 dgrad | fp64 | rel. L2 ≤ 1e-6 vs same-input reference |
| E1–E4 | fp64 | rel. L2 ≤ 1e-6 |
| R1 router | fp32 CPU | identical top-8 sets and order; weights ≤ 1e-6 rel. |
| A2/A3 attention | A4 blocked (fp32) and fp64 CPU on small shapes | fwd rel. L2 ≤ 5e-4, bwd ≤ 2e-3 (bf16 inputs) |
| D4 DeltaNet fwd | fp32 sequential (D7, mirrors llama.cpp) | rel. L2 ≤ 1e-4 per layer-window; fp64 ≤ 1e-5 |
| D6 DeltaNet bwd | fp64 sequential backward and finite differences | rel. L2 ≤ 1e-3 |
| L1 CE | fp64 | loss rel. ≤ 1e-6; `dh` rel. L2 ≤ 1e-5 |
| O1 AdamW | fp64 scalar loop | rel. ≤ 1e-6 |

**Test shapes always include**: real shapes, tile-boundary ±1, `M_e ∈ {0, 1, 7, 8, 63, 64, 65}`, all tokens to one expert, one token per expert, documents of length 1, 63, 64, 65 and 8192, a window of exactly one document, and random garbage in pad slots. **Every unit test also runs under `compute-sanitizer` (memcheck, racecheck, initcheck, synccheck)** once per release.

---

## 7. Training algorithm

### 7.1 Baseline recipe = v0.2 parity (so the first run is an apples-to-apples comparison)

| Setting | Value | Label |
|---|---|---|
| Optimizer | AdamW, β = (0.9, 0.999), ε = 1e-8, weight decay 0.0 | READ-SRC (`train.py:214`) |
| Gradient clip | global norm 1.0 | READ-SRC (`train.py:269`) |
| Schedule | linear warm-up over 5 % of total steps, then cosine to 0 | READ-SRC (`train.py:219–225`) |
| LR | 1e-4 (run 1), 5e-5 (second phase) | MEASURED/READ-SRC |
| Epochs / seed | 2 / 3407 | READ-SRC |
| LoRA | rank 16, α 32 (scale 2), dropout 0.0, 310 modules (10×4 attention, 30×5 DeltaNet, 40×3 shared expert) | READ-SRC |
| Documents per optimizer step | 4 (v0.2: token budget = max length ⇒ batch 1, accumulation 4) ⇒ ≈ 4.9 k processed tokens/step | DERIVED |
| Max sample length | 8,192 tokens; longer is a hard failure | author policy |

"Parity mode" reproduces this exactly (4 documents per step, K = 1 window). It is chosen for run T13-A so that any difference from v0.2 is attributable to the trainer, not to the recipe. Other recipes (larger steps, rank, LR) are ablations (7.11).

### 7.2 LoRA

* **Parametrization.** For a target `W (N×K)`: `y = W x + s·B(Ax)`, `A ∈ ℝ^{r×K}`, `B ∈ ℝ^{N×r}`, `s = α/r` (rsLoRA option: `α/√r`).
* **Initialization** (as PEFT's default, so v0.2 and the new trainer start alike): `A ~ Kaiming-uniform(a = √5)` drawn from the counter-based RNG indexed by (tensor id, element), `B = 0`.
* **Precision.** fp32 masters; bf16 copies used by the GEMM epilogues; the copies are refreshed after every optimizer step.
* **Targets** (D3): the 310 v0.2 modules; **MTP** modules when D12 is on (attention ×4 and shared expert ×3 in `blk.40`, optionally `eh_proj`).
* **Options decided by ablation, all off by default:** LoRA on `lm_head` (a trained `lm_head` must be rewritten: +117 MiB if raised from Q6_K to Q8_0, 0 if trained through the Q6_K quantizer in Mode Q; either way it spends the byte budget of 16.5 B2); rank 32/64 (+0.32/+0.96 GiB of states); trainable shared-expert gate vector (40 × 2048 = 81,920 params) and trainable norms (≈ 175 k parameters, ≈ 0.7 MiB, DERIVED); router LoRA (the GGUF format supports it, but changing routing invalidates the engine's static hot bank — default off); DoRA (parked).
* **GGUF mapping.** Each LoRA module is named after its GGUF tensor: `blk.N.attn_qkv.weight → blk.N.attn_qkv.weight.lora_a / .lora_b`. The adapter file carries `adapter.lora.alpha` = α (or α·√r for rsLoRA). Layout follows `convert_lora_to_gguf.py` (PEFT-style `A: r×in`, `B: out×r`). Verified by loading in llama-server (P-X3).

### 7.3 Loss and normalization (D8)

```
L_step = ( Σ_{t ∈ step} w_t · CE_t ) / ( Σ_{t ∈ step} w_t )
```
`w_t = 1` on every token the model must generate when served (7.2/8.4), `0` elsewhere; the sum runs over **all windows of the step**, so a 40-token reply and a 3,500-token reply contribute in proportion to their token counts (fixes F3). Optional per-span weights (e.g. down-weighting reasoning boilerplate, up-weighting final-answer tokens) are supported by the data schema but are a *data-policy* choice, off by default, and ablated (P-T6). If MTP is on: `L = L_main + λ·L_mtp` with the same normalization.

### 7.4 One step, concretely

1. `packer` emits the step: documents, windows, `cu_seqlens`, label weights, positions.
2. Embedding gather → forward sweep → head/CE (+MTP) → backward sweep → LoRA gradients accumulated in fp32 (5.6).
3. `O2` global grad-norm; **non-finite ⇒ stop with a report** (state dump, last tape, last 3 steps' metrics). A `--skip-nonfinite N` option exists for long unattended runs but is off by default, because a skipped step hides numerics bugs.
4. `O1` AdamW with the schedule value for this step.
5. Metrics (step-wide), canary tick (15.3), checkpoint if due.

### 7.5 Determinism and reproducibility (P7)

| Source of nondeterminism | Handling |
|---|---|
| Data order | seeded Philox shuffle per epoch over document ids; the order is a pure function of (seed, epoch) |
| Float atomics | none in `deterministic` mode; fixed-order tree reductions for LoRA gradients, grad-norm, loss, combine |
| Expert dispatch order | stable counting sort (token order inside each expert) |
| Dropout / sampling | none (dropout 0) |
| Resume | the checkpoint stores step, data cursor, RNG counters; **resume must be bitwise-equal** to an uninterrupted run (P-T11) |
| Hardware | overclocked, no ECC ⇒ canaries (15.3); determinism tests double as fault detectors: a mismatch between two identical runs is a hardware fault until proven otherwise |

### 7.6 MTP co-training (D12, conditional on P-S5)

The engine's MTP speculation (draft 2/3: 51.5–52.9 vs 45.4 tok/s at 12 k tokens, MEASURED) depends on the MTP block predicting the *next-next* token from the trunk's hidden state. After LoRA changes the trunk, an untouched MTP head is out of distribution for its input. The plan: run `blk.40` in the step, with its own LoRA, and add `λ·L_mtp` (λ = 0.1–0.3 to be ablated; DECISION pending P-T8). Position alignment (proposal: positions `t, t+1 → t+2`, masked across document boundaries; to be confirmed against the llama.cpp MTP graph, P-S5). Cost ≈ one extra layer in each sweep (≈ 2.5 %) plus one more expert-layer fetch per pass. **Acceptance-rate measurement before/after** (P-T8) decides whether it stays on.

### 7.7 Evaluation inside the trainer

`pelias-train eval` runs teacher-forced loss on the held-out split (68 rows in v0.2) with exactly the training forward path; it also writes the compact reference-logit file used by the export gate (9.6). Generation-based behaviour tests are run on the **exported GGUF with llama-server** (outside the trainer) — they exercise the real serving stack.

### 7.8 Checkpoints — four registries (adopted from the proposals)

| Registry | Contents | Count assert (derived from `ModelSpec`, not hard-coded) |
|---|---|---|
| R1 adapters | LoRA `A`,`B` fp32 masters for trunk modules | 310 (v0.2 parity) |
| R2 shared gates | `ffn_gate_inp_shexp` vectors if trainable | 40 (+1 MTP) |
| R3 norms | norm weight vectors if trainable | per option |
| R4 MTP | `blk.40` LoRA + `eh_proj`/norm deltas if trained | per option |
| plus | Adam `m`,`v`, step, schedule state, RNG counters, data cursor, config hash, base-GGUF SHA-256 | |

File format: a flat binary per registry with a JSON manifest (names, shapes, dtypes, offsets, SHA-256 of each blob) — trivially readable from Python tests. Writes are atomic (write to `.tmp`, `fsync`, rename), keeping the last three. The count asserts are derived from the spec so that the MTP block (which adds expert tensors to the file: 82 expert tensors, not `2 × 40`) cannot trip a hard-coded `2·L` check (the trap in the reviewers' bug 9).

### 7.9 Metrics (step-wide; v0.2 logged only the last micro-batch)

`loss`, `tokens`, `assistant_tokens`, `grad_norm`, `lr`, `step_s`, `tok_per_s` (processed tokens / step time), `expert_bytes`, `fetches`, `stream_wait_s`, `gpu_busy_frac`, `peak_vram_gib` (reset once per step, read from the governor and `cudaMemGetInfo`), `host_rss_gib`, `canary_ok`, plus a per-layer histogram of expert selection counts from the tape. JSONL, one line per step.

### 7.10 What each training run costs

At 1,200–1,500 tok/s (target, DERIVED) a parity run (≈ 3.0 M processed tokens) takes 33–42 min; at the minimum viable 600 tok/s, 84 min; v0.2 took 202.7 min. This is what makes an ablation programme affordable.

### 7.11 Ablation programme (each row is one run; decisions are made by evaluation, not argument)

| ID | Question | Variants | Decision metric |
|---|---|---|---|
| T-A1 | Does the new trainer reproduce v0.2's training dynamics? | parity recipe | eval-loss curve vs v0.2's (P-T9) |
| T-A2 | Forward mode | R vs F | KL vs llama.cpp; eval loss; speed |
| T-A3 | Step size | 4 docs vs 16 k vs 32 k tokens (LR scaled √) | eval loss at equal tokens; wall time |
| T-A4 | Rank | 16 / 32 / 64 | eval loss; core merge error |
| T-A5 | Gates and norms trainable | off / gates / gates + norms | eval loss |
| T-A6 | Span weights | uniform / tiered | held-out format gates |
| T-A7 | MTP | off / on (λ = 0.1, 0.3) | MTP acceptance in the engine |
| T-A8 | LoRA on lm_head | off / on | eval loss; export size |
| T-A9 | Expert-subset LoRA (ESFT-style, a few layers) | off / on | eval loss; engine cost — only if A1–A8 leave a gap |
| T-A10 | Locality-aware router training (Mode L, 7.12) | C0 vs λ_res sweep, CE-only control, two router lrs | G-L1…G-L4 |
| T-A11 | Merge-aware training (Mode Q, 4.10) | LoRA-then-merge vs Mode Q, at the served type and at Q6_K, per core class | post-export held-out CE through llama.cpp; step overhead (P-E8) |
| T-A12 | Bank-aware variants (16.5 B1): zero-miss hinge on target layers (L-Z); clamped-forward absorb then router align (L-C); distillation anchor | plain Mode L vs +hinge vs L-C; λ_zm, δ, k_T from the B1.1 table; two router lrs | G-Z1…G-Z4 (16.5 B1.5) |
| T-A13 | Bimodal-faithful forward and outlier regularizer (Mode B / N, 16.5 B5) — only if P-E12 passes | off / B / N / B+N | CE; K3 on the engine's test set |

### 7.12 Locality-aware router training — Mode L (D21; optional, gated, off by default)

**Why it is on the table.** The engine's decode speed rises with the hit rate *h* of its static hot bank in VRAM (Chapter 2.7). Measured: live *h* = 0.775–0.779 at the 7.0 GiB bank (0.806 simulated held-out); the simulated static-bank curve is 0.43 / 0.54 / 0.63 / 0.70 / 0.74 / 0.76 / 0.85 at 2 / 3 / 4 / 5 / 5.6 / 6 / 8 GiB; a least-squares line through the engine's bank-size sweep (its 8.6 table, short context) gives ≈ +1 tok/s per +0.01 *h* (DERIVED; single runs there differ by up to 1.4–1.8 tok/s between launches, so this is a rough slope, not a result). A router that sends more of its top-8 into the resident set is therefore worth the same as a bigger bank: lifting *h* by 0.027 buys back the 0.59 GiB the Q8_0 core takes out of the bank (DERIVED, interpolating the curve at 7.0 → 6.41 GiB).

**What was ruled out before, and why that ruling does not transfer one-for-one.** Forcing locality *at inference* on a frozen model (clamping or biasing the router toward cached experts) was ruled out on the strength of arXiv 2608.18261. This session the paper was read in full (the earlier citation was second-hand). What it says, and what it does not:

| Source | Finding | Label |
|---|---|---|
| arXiv 2608.18261 (*Cacheable by Design?*), §7.1, Table 6 | Training-free cache-aware rerouting on an **untrained** 137 M MoE: ≈ 30 % fewer misses cost +0.8 % to +2.0 % perplexity depending on seed; 50 % fewer cost +4.0 % to +5.8 %. This is the result the engine blueprint relied on, and it holds. | PUBLISHED |
| same, Table 5, §7 | **Training** routers for locality from scratch (137 M and 340 M parameters, 200 M tokens, Switch load-balance loss): miss reduction and perplexity cost are coupled; no loss weight meets "≥ 30 % fewer misses at ≤ +1 % perplexity". Pre-registered, refuted. λ = 0.03: 24 % reduction, +1.8 %; λ = 0.05: 59–60 %, +2.1–3.1 %. | PUBLISHED |
| same, §7.1 stacking | On a locality-**trained** model the same rerouting step (τ 0 → 0.5) costs **+0.9 %** instead of **+7.7 %** on the baseline. Training absorbs most of the clamp's cost. This supports the idea that inference-time restrictions can be absorbed in training. | PUBLISHED |
| same, §8–9 | The locality "tax" went from +2.0 % (137 M) to +2.5 % (340 M); the authors call the scale trend "suggestive" only. Limits stated: two small sizes, under-trained, one GPU. **Nothing at 35 B / 256 experts.** | PUBLISHED |
| ReMoE, arXiv 2605.27081, §4–5 | **Router-only** fine-tuning of the pretrained DeepSeek-V2-Lite (64 experts, top-6) with **frozen experts**: adjacent-token expert reuse 27.3 % → 34.5 %; MMLU +0.09, HumanEval +2.44, GSM8K −0.76, IFEval −1.11 points (the authors: within uncertainty). A CE-only router fine-tune **lowers** reuse (0.2293 vs 0.2730 baseline), so the locality term is what does the work. Loss = CE + λ_KL·KL-to-frozen-router + α·reuse/smooth/lag/working-set terms. Load imbalance rises (CV 0.041 → 0.161), accepted for batch 1. | PUBLISHED |
| arXiv 2609.04895 (*Cache-Aware Joint Router Adaptation*), abstract only | Post-training adapts the backbone jointly with small auxiliary routers; native top-K kept; reports hit rate and traffic, no speed. | PUBLISHED (abstract read) |
| arXiv 2607.08780 (*Sticky Routing*) | Cited by 2608.18261 as reporting near-free locality losses on smaller single-domain models. The PDF could not be read this session. | UNVERIFIED |

**Reading.** The honest summary is: untrained clamping costs real quality beyond ≈ 30 % miss reduction; training from scratch at tiny scale did not beat the +1 % bar; but (a) *post-training the router of a pretrained model with the experts frozen* is exactly ReMoE's setup and ours, and (b) stacking shows training makes the remaining clamp cheap. Whether a 256-expert, top-8, 4-bit-frozen model behaves like either is **unknown**; Mode L is therefore a gated experiment with a pre-declared pass bar, not a promise. (DECISION)

**What the author's setup adds.** Four things differ from the papers, two in our favour and two against.

* *For:* the objective is a **static** bank, so the target is a linear functional of router probabilities (mass on a known set R_l per layer), more direct than adjacent-token similarity; and 21.2 M LoRA parameters in the core (attention, DeltaNet, shared experts) can move the hidden states that the frozen experts see.
* *For:* the served router is **F32**, so a trained router merges *exactly* (no requantization step, unlike the core).
* *Against:* the experts are frozen **4-bit** bytes; tokens routed to new experts meet specialists that cannot adapt. Each expert is small (FFN 512), so substitutes may be poor.
* *Against:* the training set is ≈ 1.2 k documents. A router tuned to them could lose hit rate on other domains (engine domain rates: reasoning 0.674 < code 0.753 < agent 0.790 < persona 0.799 at the 6.1 GiB bank). ReMoE itself did not measure out-of-distribution domains.

**Design.**

1. **Trainable set.** `ffn_gate_inp` (W_r, 2048 × 256 per layer, F32; 41 tensors with the MTP block; 21,495,808 parameters; DERIVED) — full fine-tune at fp32 (default, "form A"). Alternatives (ablation only): rank-8 LoRA on W_r (needs P-L9 to show llama.cpp applies an adapter to the router tensor), or a per-layer logit bias (10,240 parameters; needs an engine/llama.cpp graph change, because the served graph has no bias tensor — the P-L2 oracle uses exactly that change).
2. **Loss.** L = L_CE (+ MTP) + λ_trust · KL(p ‖ p⁰) + λ_res · L_res, where p = softmax over **all 256** router logits (fp32), p⁰ the same for a frozen copy of the served W_r, and

   L_res = − Σ_l w_l · mean_t log( Σ_{e ∈ R_l} p_l,t(e) + ε ),  w_l = (expert bytes of layer l) / (total),

   averaged over **assistant-token positions only** (decode is where the bank matters; prompt tokens still receive the trust term). R_l is the per-layer resident set the engine would pick for its reference bank (from its hot-set file or from this trainer's tape at the target bank size). An optional λ_reuse term (ReMoE's reuse/smooth) is kept for dynamic-cache variants of the engine; it is off here because the engine's bank is static.
3. **Gradient path.** The CE gradient reaches W_r through the renormalized top-8 weights (selection itself is not differentiated — the same as 6.3); the locality and trust terms act on the full 256-way softmax, so they also move the logits of unselected experts, which is how membership changes. The routing tape is re-recorded every step; the backward recompute uses the step's own tape (D7), so a changing router never makes a step inconsistent.
4. **Metrics logged every step (free from the tape).** Top-8 hit fraction on R_l (the true objective, non-differentiable), surrogate mass on R_l, mean KL to p⁰, expert-usage CV, adjacent-token overlap. P-L3 checks that the surrogate tracks the true hit fraction; if it does not, the surrogate is replaced (e.g. by the mass of the renormalized top-8 set).
5. **Schedule.** Phase 0: normal run, Mode L off → checkpoint **C0** (always kept, always exported: it is the quality reference and the fallback). Phase 1: short continuation from C0 with Mode L on, router + core jointly trainable, lr_router sweep starting at 2×10⁻⁵ (DECISION as a starting value only; ReMoE's 5×10⁻⁵ was for a different model), λ_res sweep in P-L4 — each λ is a separate short run from C0, so the sweep never contaminates C0. Phase 2: re-profile with the trained router's tape, recompute R_l; if mean Jaccard(R_l, R_l′) < 0.9 repeat Phase 1 once with the new R_l (at most two rounds). Phase 3: export every candidate (9.3 rule below) and let the engine's measured speed and the quality gates choose.
6. **Cost.** FLOPs: the W_r weight-gradient is 2·2048·256 per layer per token = 41.9 MFLOP per token over 40 layers = **0.21 %** of the 20.39 GFLOP step (DERIVED). VRAM: trainable copy 82 MiB + gradient 82 MiB + Adam 164 MiB = **+0.32 GiB** with the frozen reference kept on device (DERIVED; inside the 3.0 GiB headroom of 5.3). No change to host memory or PCIe traffic. The router weight-gradient reduction is deterministic (O3 pattern).

**Gates (DECISION, declared before any run so the result cannot move the bar).**

| Gate | Pass condition | Measured by |
|---|---|---|
| G-L1 quality | held-out assistant-token CE of candidate ≤ 1.01 × CE(C0), **and** no regression on the author's generation checks | `eval` (7.7), P-L10 |
| G-L2 benefit | simulated held-out static-bank hit rate at the engine's reference bank ≥ C0's **+ 0.03** (about what an all-Q8_0 core would cost the bank: 0.027 for +0.59 GiB, DERIVED above; the gate is kept at 0.03 so Mode L must pay for a substantial core upgrade) | tape + engine simulator (P-L1/P-L4) |
| G-L3 breadth | hit rate in each domain class ≥ C0's − 0.01 on classes **not** in the training set | P-L6 |
| G-L4 engine | decode tok/s above C0's by more than the run-to-run noise (3 launches per side) at the engine's reference cell | P-L7 |

If no (λ, lr) pair passes G-L1 and G-L2 together at two router learning rates, Mode L is closed with the negative result recorded, as the paper did; the code path stays off and costs nothing.

**Export and hand-off.** The router tensors become a second class of rewritten tensor next to the core (rule added to 9.3); everything else remains verbatim and hash-proven. The exact reference for a Mode L model is *base-with-new-router* (F32 router spliced into the served base, zero requantization) + the adapter GGUF — emitted as a third artifact. The engine's hot set must be re-chosen from the trained model's tape (9.5). An inference-time bias on top of a Mode L model (the stacking result above) is an **engine-side** option, handed to the engine cross-check; it is not a trainer feature.

**Probes:** P-L1 … P-L10 (Chapter 11.8). P-L1 and P-L2 need no trainer code and run in Wave 0, so the size of the prize and the cost of the untrained clamp on this exact model are known before any of the above is built.

---

## 8. Data pipeline

The dataset is the author's private data; it never enters this document or the repositories. This chapter specifies the machinery.

### 8.1 Input schema

JSONL; one conversation per row: `messages[]` of `{role, content, reasoning_content?, tool_calls?, tool_call_id?}` with roles `system`, `developer`, `user`, `assistant`, `tool`; optional per-row `family` tag and per-span weights. Anything else is rejected by the audit with the row index.

### 8.2 Tokenization — the served tokenizer, not a lookalike

The vocabulary and merges are read **from the GGUF itself** (`tokenizer.ggml.*`, model `gpt2`) through libllama's vocab API, with special-token parsing on exactly as in llama-server. Parity target: identical token ids to the HF tokenizer on the whole dataset (**P-D1**) — checked in the Python oracle only; the product never imports it.

### 8.3 Chat template — rendered by llama.cpp's own engine

`tokenizer.chat_template` from the GGUF is rendered with llama.cpp's jinja implementation (`common/jinja`, `common/chat`), i.e. the code path llama-server uses. Known traps handled explicitly (older rundowns, READ):

* the template splits assistant text on `</think>`: a literal `<think>`/`</think>` inside an answer is silently removed in 82 of 3,000 turns when `reasoning_content` is not supplied — the pipeline always passes reasoning separately and **round-trip checks** every assistant turn (render → tokenize → detokenize → compare to source; **P-D2**);
* tool calls use the native XML form (`<tool_call><function=…><parameter=…>`), rendered by the template, never by hand;
* the `developer` role exists in the newer template;
* user text containing a literal special token (`<|im_start|>` …) would be tokenized to the special id by `parse_special`: the audit flags it rather than letting it through.

### 8.4 Label masks — "train on exactly what the model must generate when served"

For each assistant turn, render the prompt with `add_generation_prompt = true` and the full conversation; the tokens after the common prefix, through and including `<|im_end|>`, are the trainable span; everything before (system, user, tool results, **and the generation-prompt tokens including the opening `<think>` marker the server adds itself**) has weight 0. Multi-turn rows: every assistant turn is trained by default (`--train-on all`), or the last only. Verified independently in **P-D5** by re-deriving spans from marker tokens and comparing token by token.

### 8.5 Packing (D10)

1. Seeded shuffle of documents (per epoch).
2. Form a step: parity mode = next 4 documents; token mode = documents until the token target (default 16,384) or the document cap; never fewer than the step minimum (3,072 tokens) except the last step of an epoch.
3. Round each document up to a multiple of 64 slots; best-fit-decreasing into windows of ≤ 8,192 slots (window sizes snapped to {2048, 4096, 6144, 8192}).
4. A single document longer than 8,192 slots is a **hard failure** listing the offending row ids (never truncated). Policy note from the older rundowns: 4,096 would cut every DAG-style sample (≈ 5–7 k tokens), hence 8,192.
5. Emit: `token_ids`, `positions` (restart at 0 per document), `cu_seqlens`, `weights`, pad mask.

**Equivalence test (P-D3):** forward loss and LoRA gradients of a packed window equal those of the same documents run one by one, within the rung-1 tolerances (this is what proves the attention, DeltaNet and conv masks are right).

### 8.6 Audits (all run before any GPU time; any failure stops with a report that contains ids and counts only)

| Audit | Check |
|---|---|
| A-LEN | histogram of document lengths; count over 8,192 (must be 0); chunk-alignment padding waste (P-D4) |
| A-TOK | token-id parity with the oracle on every row |
| A-TPL | template round-trip on every assistant turn |
| A-MASK | span derivation parity (P-D5); no empty assistant span; the EOS token is inside the span |
| A-DUP | normalized-text hash duplicates within train and between train and eval (P-D6) |
| A-SPECIAL | user content containing special-token text |
| A-FAMILY | family counts match the declared mix (for the "≥ 75 % reasoning rows" style rules, if used) |

---

## 9. Export

### 9.1 Outputs of one run (always written to a new, timestamped directory; the served GGUF is never overwritten)

| Output | Purpose |
|---|---|
| `adapter.gguf` | GGUF LoRA adapter — **zero-drift reference**; also directly servable with `--lora` |
| `merged-core.gguf` | the production candidate: every unchanged tensor byte-identical to the base, the core at the budgeted per-class types of 9.4 |
| `export_report.json` | base SHA-256, per-tensor SHA-256 table (unchanged tensors), core types and sizes, merge error statistics per class, KL results, tool versions |
| `expert_usage.json` | per-layer, per-expert selection counts accumulated from the tape over the training set (hot-bank re-profile input for the engine) |
| `ckpt/` | final four-registry checkpoint |

### 9.2 The adapter GGUF

Written per `convert_lora_to_gguf.py` conventions: `general.type = adapter`, `adapter.type = lora`, `adapter.lora.alpha`, tensors `<gguf tensor name>.lora_a` / `.lora_b`. Loaded by stock llama-server (`--lora`), applied at run time by `build_lora_mm` (scale = alpha/rank). Because it applies the exact trained delta to the exact base bytes, KL(adapter-served ‖ trainer) is limited only by the serving arithmetic — the cleanest check of the *trainer* (P-X3).

### 9.3 The merged-core GGUF — byte-splice

1. Open the base GGUF read-only (memory-mapped). Parse tensor infos.
2. For each **core** tensor `W`: stream tiles; `W_eff = dequant(W_base) + (α/r)·B·A` in fp32 (dequantization by the G8 reference; the delta by an fp32 GEMM); quantize to the class's target type with ggml's reference quantizer (Q8_0: `d = amax/127`, `q = round(x/d)`; Q6_K and Q4_K: the reference search ported from `ggml-quants.c`, MIT); store `d`, `dmin` as f16 and the packed 6-bit scales exactly as ggml does. Bit-exactness of each quantizer against ggml is probe P-E6 (it extends P-Q1, which covers dequantization). In Mode Q the bytes are those of the last `W_eff` (4.10) and this step is a copy.
3. Tensors with trainable norms/gates: F32 values from the registries.
   **Mode L only (7.12):** the 41 router tensors (`ffn_gate_inp`) are also rewritten, as F32 from the router registry; they are verified equal to the checkpoint, not to the base.
4. Rewrite the header: new tensor infos (type, size, offset) for changed tensors; all other infos unchanged except offsets; alignment 32; all KV metadata copied; add `pelias.*` provenance keys (trainer version, base SHA-256, run id, core type) — ignored by llama.cpp; `general.file_type` left as in the base.
5. Write the data section: unchanged tensors **copied verbatim** (streamed from the mapping); changed tensors from step 2–3.
6. **Proof:** after writing, re-open and hash every unchanged tensor payload; the report lists `unchanged: N tensors, identical: N`. In Mode L the list of changed tensors is core + routers; the report names both classes and their counts. A mismatch aborts and deletes the partial file.
7. Space gate before starting: the export needs ≈ 21 GB free on the destination drive (the older export scripts' `need` check, kept); the destination is never the drive holding the served GGUF unless space is verified.

### 9.4 Core type assignment — a byte-budget solver (D22)

**Inputs.** (1) The served type of every core tensor (READ-SRC). (2) The engine's supported weight types (P-E5; today `Q4_K`, `Q6_K`, `F32`; `Q8_0` only if the engine adds it). (3) The byte budget `Δ_max` = ceiling (10.7 GiB) − the engine's measured peak at the chosen bank size and context − a 0.1 GiB margin (P-E4; DERIVED at the 7.5 GiB bank from the engine's measured whole-card peaks: **0.16 GiB at the default micro-batch** (peak 10.40–10.44 GiB) and **none at `-ub 2048`** (peak 10.67–10.69 GiB; −0.09 GiB after the margin). If the ceiling is read as a limit on the process's own allocations the same peaks leave ≈ 1.2 / 0.9 GiB (16.3.1, decision 7). `Δ_max` is therefore a function of the engine's prefill micro-batch; the solver is run for the micro-batch the engine ships with). (4) The retention table of P-Q3 per class and type.

**Class sizes (MiB, DERIVED from the real inventory).**

| Class | Params | Served mix | All Q6_K | All Q8_0 |
|---|---|---|---|---|
| Shared experts | 125.8 M | 72.7 | 98.4 | 127.5 |
| DeltaNet in (qkv + gate) | 755.0 M | 462.8 | 590.6 | 765.0 |
| DeltaNet α/β | 3.9 M | 2.1 | 3.1 | 4.0 |
| DeltaNet out | 251.7 M | 135.0 | 196.9 | 255.0 |
| Full attention (q, k, v, o) | 272.6 M | 147.8 | 213.3 | 276.2 |
| **Total** | **1,409.0 M** | **820.3** | **1,102.3 (+282)** | **1,427.7 (+607)** |

**Rule.** Choose a type per class (tensor-level only if P-Q3 shows it matters) to maximize the predicted retention subject to `Σ bytes − served ≤ Δ_max`, using only supported types; ties go to the smaller. Examples at the whole-card reading of the 10.7 GiB ceiling (16.3.1; it was 369 MiB and ≈ 113 MiB at 10.9). The per-class raises to Q6_K cost +25.7 (shared experts), +65.5 (attention), +61.9 (DeltaNet out), +127.8 (DeltaNet in), +1.0 (α/β) MiB. At `Δ_max = 164 MiB` (default micro-batch): all-Q6_K (+282) misses by 118 MiB; shared experts + attention + DeltaNet out (+153.1) fit with 10.7 MiB spare; DeltaNet in + shared experts + α/β (+154.5) fit with 9.3 MiB spare; attention + DeltaNet in (+193.3) misses by 29.5 MiB; the classes left out stay at their served types, closed by Mode Q (4.10). At `-ub 2048` there is no budget (−92 MiB after the margin): every class stays at its served type and is closed by Mode Q, or bank is given up (16.3.1). On the process-only reading (≈ 1.2 / 0.9 GiB) both all-Q6_K and the all-Q8_0 core (+607 MiB) fit at both micro-batches (the engine would still need Q8_0 kernels, P-E5). Which classes get the scarce bytes is decided by the retention table of P-Q3, not by this arithmetic. Classes that fail P-Q3 at every affordable type are handled by Mode Q (4.10) at their served or Q6_K type. The chosen table is written into the report and the GGUF key `pelias.core_types`.

### 9.5 Engine hand-off

1. Load `merged-core.gguf` in llama-server and in the pelias-engine tier build; run the engine's smoke and KL gate (P-X5).
2. Re-profile the hot bank from `expert_usage.json` (the trained model's routing differs from the base's; domain hit-rates were 0.67–0.80 and layers 0–2 are the weakest). For a Mode L candidate this step is part of the training loop (7.12, Phase 2), and the candidate is judged by the engine's measured hit rate and tok/s against C0 (G-L2, G-L4).
3. Re-measure decode and peak VRAM at the engine's reference cell with the real merged core (P-X5, P-E4): the core's extra bytes come out of the bank or the headroom; the regression budget is −2 % (ESTIMATE). If peak VRAM exceeds the 10.7 GiB ceiling at the 7.5 GiB bank the assignment is rejected and the solver is re-run with a smaller `Δ_max`.
4. Windows-native and WSL2 are both re-measured (the engine found +24–28 % decode on Windows-native).

### 9.6 The export gate — how "faithful" is decided

* **Reference logits:** during `eval`, for each held-out position with weight > 0, the trainer stores the top-256 log-probabilities and the log-sum-exp of the remainder (≈ 131 MB for 64 k positions), computed with its own forward path.
* **Candidate logits:** the exported GGUF is loaded through libllama in the `verify` tool (a separate process run after training, so the pinned store is not alive at the same time) and run teacher-forced on the same tokens.
* **Metrics:** truncated KL (top-256 + lumped tail), top-1 agreement, mean |Δ log-prob| on the target tokens.
* **Thresholds:** set from the llama.cpp float-spread measured on the same tokens (P-Q4): **pass iff KL(exported ‖ trainer) ≤ 1.0 × the stock-vs-stock batch spread** (the author's H6 rule: the float spread governs). Both the adapter export (expected ≈ spread) and the merged export are measured; the *difference* between them is the merge error and is reported separately.

---

## 10. Verification ladder

A trainer is only as good as the evidence that it computes what it claims. The ladder below is built so that **every rung is testable against something already trusted**, and no rung is attempted before the one under it passes. Nothing in the ladder needs the dataset except rungs 6 and 9, and those report counts and pass flags only.

### 10.1 Three oracles, and who wins a disagreement

| Oracle | What it is | Strength | Weakness | Lives in |
|---|---|---|---|---|
| **A — llama.cpp** (commit `2ca15f540`, run through libllama and `eval-callback`) | The served truth: the exact arithmetic that decides what users see | It *is* the target; per-layer tensors can be dumped by name (P-S1) | Not a pure function: CPU and GPU paths differ (P-Q4, P-S6), the fused GDN op differs from the unfused graph (P-S7); has no backward pass | `tools/oracle_llama/` (links libllama; test-only) |
| **B — fp64 reference** | Our own CPU implementation of every op, forward and backward, in `double` (sequential DeltaNet, materialized attention, plain loops) | Unambiguous; the backward formulas are derived once (6.7) and checked by finite differences | Slow (small shapes and 2–4 layer slices only); written by us, so it can share our misreadings | `tests/ref/` |
| **C — PyTorch fp32 autograd** | A throw-away HF-style model of the same slice, with autograd | Independent derivation of the backward; catches formula errors in B | Python; different conventions (norm +1, V-head order, MoE routing code) need the inverse permutations; known version-specific traps (Appendix C) | `tools/oracle_torch/` — **outside the product tree, never imported by it** |

**Who wins.**

1. *Forward, served semantics:* **A wins.** If B disagrees with A on a deterministic quantity (a conventions question: norm, RoPE pairing, tap order, tie-breaking), B is wrong until shown otherwise. If A disagrees with itself across its own legal float variants, the disagreement is the **float spread** (P-Q4) and defines the tolerance, not a bug.
2. *Backward:* there is no A. **B is checked against finite differences in fp64 first** (a gradient is correct if the directional derivative matches to 1e-8 relative on a tiny config); C is then used as an independent cross-check. If B and C disagree and finite differences side with one, that one wins and the other is fixed.
3. *Numerics of our fast kernels:* judged against B on the **same inputs** (not against A), so a kernel is not blamed for the served model's own noise (6.11).

### 10.2 The ten rungs

| Rung | Question | Evidence (oracle) | Gate to climb | Probes / tests |
|---|---|---|---|---|
| **0 Platform** | Is the machine stable and the build reproducible? | Wave 0 records; clean-checkout build on WSL2 and Windows | No Xid, canary hashes equal (P-H4, P-H10); clean-checkout build run twice, identical test results (exit gate of stage T1, Chapter 12) | P-H1–H12, P-M3, P-M4 |
| **1 Kernels** | Does each kernel equal its reference on every awkward shape? | B and ggml reference code | Tolerances in 6.11; `compute-sanitizer` clean | P-Q1, P-Q2, P-K-series |
| **2 Components** | Does one attention / DeltaNet / MoE block match the served block? | A (per-layer dumps), B | rel. L2 within 2× the llama.cpp CPU-vs-GPU difference for the same block (P-Q4 supplies it) | P-S1–S8, P-Q7, P-Q12 |
| **3 Model forward** | Does the whole forward give the served logits? | A | Truncated KL ≤ 1.0 × float spread on 64 held-out documents, mode R and mode F both reported; per-layer error profile flat (no layer jumps) | P-Q5, P-Q6 |
| **4 Gradients** | Is every gradient right? | B by finite differences; C | Directional derivative match; LoRA-gradient cosine ≥ 0.99999 vs C on a 4-layer slice | P-T1, P-T2 |
| **5 System invariants** | Does scheduling change the answer? | B, repeat runs | Packed = unpacked; K = 1 and K = 4 windows equal; two seeds-equal runs bitwise-equal; zero allocations in the step loop; PCIe stall ≤ 5 % | P-D3, P-T4, P-T5, P-T7, P-M5 |
| **6 Training semantics** | Is it training on the right tokens with the right loss? | A (template/tokenizer), v0.2 ledger | Zero tokenizer/template/mask mismatches; learning-curve parity with v0.2; overfit works; resume is bitwise | P-D1, P-D2, P-D5, P-T3, P-T9, P-T11 |
| **7 Export** | Does the file contain exactly what was trained? | A (loads the file) | Unchanged tensors hash-identical; KL(file ‖ trainer reference) ≤ 1.0 × float spread; adapter-served KL likewise | P-X1–X4 |
| **8 Serving** | Does the engine still run it as fast as before? | pelias-engine benchmarks | Decode regression within the declared budget; hit rate re-profiled | P-X5, P-L7 |
| **9 Behaviour** | Does it behave as the author needs? | The author's private held-out checks (counts and pass flags only) | The author's declared bars; Mode L gates G-L1…G-L4 if Mode L is used | P-L10, T-A programme |

### 10.3 The seven gates (what `pelias-train verify` runs and prints)

| Gate | Covers | Pass = |
|---|---|---|
| **G1 Build** | rungs 0–1 | clean build; all unit tests; sanitizers clean |
| **G2 Parity** | rungs 2–3 | KL and per-layer thresholds above |
| **G3 Gradients** | rung 4 | finite-difference and oracle-C checks |
| **G4 Invariants & semantics** | rungs 5–6 | equivalence, determinism, template, masks, curve parity |
| **G5 Export** | rung 7 | hash proof, KL gates |
| **G6 Serving** | rung 8 | engine smoke, regression budget |
| **G7 Acceptance** | rung 9 | the author's bars; Mode L gates if used |

Each gate writes one `verify_<gate>.json` (probe ids, labels, numbers, thresholds, pass/fail, machine-state hash) and one line in `verify_summary.md`. A failed gate stops the pipeline; a gate may be waived only by a dated DECISION line in the run's `decisions.md`, signed with the reason, and the waiver is printed at the top of every later report.

### 10.4 Fixtures and privacy

* **Synthetic fixtures** (committed to the repo): public-domain text, random weights at the real shapes, 2–4-layer slices; they cover every kernel and the data pipeline logic.
* **Real-model fixtures** (never committed): per-layer fp32 dumps from the served GGUF on three fixed sequences; the 64 held-out documents' reference logits. They are regenerated by script (P-S1), hash-pinned in a manifest, and stored under the machine's results folder.
* **Dataset-derived outputs** are limited to counts, lengths, ids and pass flags. No document text, persona text or private equation appears in any report, log or commit.

### 10.5 What the ladder does not prove

It does not prove the model is *better* for the author's purpose — that is rung 9 and the ablation programme. It proves the trainer computes what the serving stack computes (rungs 1–3), that its gradients are right (4), that scheduling does not change results (5), that it trains on the right tokens (6), and that the file is exactly what was trained (7–8). Each of those is a claim the v0.2 trainer could not support (Chapter 2.6, findings F1–F9).

---

## 11. Probe catalogue

**100 probes.** Each turns a number or claim that is not yet verified into a measurement. A probe is a *specification*: a question, a method, a pass/fail rule, the file it writes, and the stage at which it can first be run. The hardware set (P-H1 to P-H10, P-H12, P-H13, P-M3, P-M4) now exists as one standalone program and its results are in 11.12; every other probe becomes a small program that writes one result file when the author says go. Anything this document marks ESTIMATE or UNVERIFIED points at one of these.

### 11.1 How to read and run them

| Column | Meaning |
|---|---|
| **ID** | `P-H` hardware/platform, `P-M` memory, `P-K` kernels, `P-Q` quantization and numerics, `P-S` served-model semantics, `P-D` data, `P-T` trainer behaviour, `P-X` export and engine, `P-L` router locality (Mode L), `P-E` engine co-design (Chapter 16) |
| **When** | **W0** = Wave 0: needs no trainer code (standalone CUDA/C++ programs, llama.cpp tools, file analysis, or a measurement the engine project owns). **T*n*** = first possible at stage T*n* of Chapter 12. |
| **Pass / fail** | The rule that turns the result into a decision. Where no rule is possible the probe says "Record" and names what the record feeds. |
| **Output** | One JSON per probe (schema in 11.11), plus raw logs. |

**Counts.** Wave 0: **47** probes (P-H13 was added on 2026-10-24 when the display-timeout test was built; three of them are the engine's own measurements; two, P-E1 and P-E9, were already run once on 2026-10-23 and are re-run on every candidate). By stage after Wave 0: T1: 1, T2: 4, T3: 6, T4: 4, T5: 2, T6: 3, T7: 2, T8: 2, T9: 4, T10: 2, T11: 4, T12: 7, T13: 2, T14: 10.

**Running rules.** (1) Launched by the author: from WSL, or on Windows-native by the one PowerShell script the author permitted on 2026-10-23 (it reads facts, builds and runs the same program; it changes no setting and installs nothing); no other use of PowerShell. (2) One probe at a time, with P-H8 (free VRAM) and the machine-state record first. (3) Results go to a new dated folder; nothing is overwritten. (4) The VRAM governor's 10.7 GiB ceiling applies to probes too. (5) Dataset-derived outputs are counts and pass flags only. (6) The HF token is never read or printed. (7) A probe that needs the author's files (the v0.2 adapter for P-Q3, the bf16 source for P-X0) asks for the path first. (8) Memory-spike check (added 2026-10-24): every run writes a GPU-memory log, `vram_log.tsv` (used and free before and after each probe); `SETTLE=1` / `-Settle` waits, before each probe, until used memory is back to the run's starting level plus 64 MiB (the P-M1 agreement size) for at most 60 s; `RETRY_ON_FAIL=1` / `-RetryOnFail` reruns a short probe that exited 2 or 3 (never P-H4, P-M3, P-H13) once plain and once after settling, tags the files `retry_plain` / `retry_settled`, and states which passed (3.3, U5).

### 11.2 Hardware, platform and memory (P-H, P-M)

These settle what the machine really does. They are standalone C++/CUDA programs or shell measurements; none needs trainer code.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-H1** | What are this card's real limits: SM count, shared memory opt-in per block, registers per SM, L2 size, copy-engine count? | Standalone program: `cudaGetDeviceProperties`; ramp `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` until it fails; read the async-engine count. | Record. Flag if opt-in shared memory < 99 KiB per block or SM count ≠ 46 (PUBLISHED expectation). Tile constants in `kern/` are derived from this file. | `h1_device.json` | W0 |
| **P-H2** | What are the real VRAM bandwidth (read, write, copy) and host-DRAM read bandwidth at the author's memory overclock? | Device streaming kernels over 2 GiB; CPU read of 4 GiB with 1/4/8/16 threads (AVX2 loads). | Record (stock VRAM spec 504 GB/s PUBLISHED; CPU 45.9 GB/s MEASURED by the engine). Feeds the roofline. | `h2_bandwidth.json` | W0 |
| **P-H3** | What are the real dense tensor rates — bf16 `m16n8k16` fp32-accumulate, int8 `m16n8k32` int32-accumulate, fp16, fp8 e4m3 — sustained, at stock clocks and at the author's overclock? | Register-resident-operand microbenchmark, ≥ 60 s per type, NVML logging of clock, power, temperature, throttle reasons. | Record; they replace the DERIVED 58.3 / 233 TOPS (stock) and 65.4 / 261 (OC) used for ε. Flag if sustained < 85 % of burst. | `h3_peak.json` | W0 |
| **P-H4** | Is the overclock stable under hours of tensor-core load (no ECC)? | 30-minute soak alternating the P-H3 kernels with a reference-checked GEMM every 100 ms; NVML throttle reasons; kernel log scanned for Xid events. | Zero output mismatches, zero Xid, no thermal throttle. Otherwise run clocks are stepped down and the step is recorded in the machine-state file. | `h4_soak.json`, log | W0 |
| **P-H5** | What are the real host→device rates for pageable, pinned (`cudaHostAlloc`), registered and zero-copy memory, by size, with 1 and 3 streams? | `cudaMemcpyAsync` 1–512 MiB with event timing; zero-copy by kernel read. | Record. The streamer assumes ≥ 26 GB/s bulk (engine zero-copy 26.1 GB/s). If pinned bulk < 20 GB/s the PCIe-hiding knee moves above ≈ 3 k tokens and the roofline is re-run. | `h5_h2d.json` | W0 |
| **P-H6** | How much do PCIe copies and CPU DRAM traffic slow each other? | H2D at full rate while 0/4/8/16 threads stream-read DRAM, and the reverse. | Record slowdown factors (engine prior: together 47.6 GB/s vs 45.9 alone). The roofline uses them. | `h6_contention.json` | W0 |
| **P-H7** | What is the largest pinned allocation this WSL2 allows (target 32 GiB), and what does the host do? | Grow `cudaHostAlloc` in 1 GiB steps to 32 GiB, touching every page; watch `/proc/meminfo` and swap. | Pass: ≥ 24 GiB, no OOM-kill, swap-in < 100 MiB. Else the fallbacks in 5.3 apply (bf16 boundary store, smaller step, pageable boundary store). | `h7_pinned.json` | W0 |
| **P-H8** | How much VRAM is really free at start (BAR1 oddity, desktop use) and after context creation? | `cudaMemGetInfo` and NVML before/after context init, with the usual Windows apps open; BAR1 total and used. | The run refuses to start unless the **smaller of CUDA free and NVML free after the context exists is ≥ the ceiling (10.7 GiB; margin 0, options `--ceiling-gib` and `--margin-gib`)**. The first draft required ceiling + 0.3 GiB (11.2), which this card cannot give: free after the context is 10.83–10.85 GiB (16.3 X16). Result recorded per run. | `h8_vram.json` (every run) | W0 |
| **P-H9** | Do CUDA graphs capture and replay correctly on this WSL2 setup (kernel and memcpy nodes, parameter update), and what is the launch saving? | 100 small kernels as stream launches vs one graph; update-in-place of kernel parameters; 10⁵ replays. | Adopt per-layer graphs only if replay saves ≥ 30 % of launch time with zero errors over 10⁵ replays. (Engine prior: graphs-off cost +4.5 ms/token in decode, a different regime.) | `h9_graphs.json` | W0 |
| **P-H10** | Is the card bitwise-repeatable (soft-error canary)? | A fixed GEMM + reduction run 10⁴ times at run clocks; hash every output. | All hashes equal. Any difference marks the clock setting unsafe for training. | `h10_canary.json` | W0 |
| **P-H11** | Do P-H3, P-H5 and P-H9 differ between WSL2 and Windows-native (the engine saw +24–28 % decode on Windows-native)? | Build the same programs with MSVC + nvcc (the permitted PowerShell script, rule 1) and run the `.exe`. **Run 2026-10-23/24 for P-H3, P-H5 and P-H9: no difference on the GPU side; the host-side costs differ (11.12).** | Record deltas; the larger is the default platform. A platform is dropped from the plan only on a measured loss. | `h11_platform.json` | W0 |
| **P-H12** | Do H2D and D2H copies overlap each other and compute (copy-engine count, driver behaviour)? | Three-stream test: H2D + D2H + a compute kernel, each alone, then together; three compute kinds (one long tensor launch, many short tensor launches, a VRAM-read kernel) × four submission patterns (copies first, compute first, interleaved; each with and without a `cudaStreamQuery` flush after every enqueue); a GPU-side event timeline per stream. | All-three overlap efficiency ≥ 0.90 for many short tensor launches with the best pattern → `S_w` and `S_x` run concurrently and the streamer uses that pattern (flush rule, 5.4). Else the solver serializes them and boundary traffic is added to the stream time (5.5). **Result: 0.92 with the flush, 0.47–0.52 without, on both platforms.** | `h12_overlap.json` | W0 |
| **P-H13** | Does a long GPU job make Windows reset the display driver (TDR), and does the work survive? | Five stages, each run only after the author types YES: one kernel per SM for 2 s, 5 s and 30 s; a queued set of kernels (10 s); a graph (10 s). Records wall time, errors, device state afterwards, NVML clocks and (Windows-native) the display-driver event log. | Pass: the stage completes, the device is usable afterwards, no errors. A failed stage stops the longer ones. Sets the longest kernel the trainer may launch without slicing it. **Result: all five stages passed on both paths, no driver events, nothing seen on screen.** | `h13_<stage>.json` | W0 |
| **P-M1** | Does the VRAM governor's accounting equal reality? | Allocate the arenas of the plan; compare the governor's sum with the `cudaMemGetInfo` delta and NVML used. | Agreement within 64 MiB. | `m1_governor.json` | T1 |
| **P-M2** | What is the real peak VRAM of the per-layer working set at T = 2 048 / 4 096 / 6 144 / 8 192 (replaces the 2.50 GiB ESTIMATE)? | One real layer forward+recompute+backward with real shapes under the arena allocator. | Peak ≤ 2.5 GiB at 8 192. If not, window sizes or arenas are re-planned (5.3). | `m2_working_set.json` | T7 |
| **P-M3** | Does WSL2 hold 28–33 GiB resident + pinned for 30 minutes under the 48 GiB cap without killing, swapping or stalling Windows? | Allocate the planned expert store + boundary store and touch continuously. | No OOM-kill, no swap-in, Windows responsive. | `m3_hostmem.json` | W0 |
| **P-M4** | How fast is the GGUF read path (≈ 20 GiB sequential) from the model drive, with and without the page cache? | Timed `read()` and `mmap` passes, caches dropped between. | Record; sets the start-up expectation and tells whether repacking vs mapping matters (the engine's open question). | `m4_read.json` | W0 |
| **P-M5** | Is the steady state allocation-free and identical step to step? | 50 steps with the allocator trace on. | Zero `cudaMalloc` after warm-up; peak identical every step. | `m5_trace.json` | T10 |

### 11.3 Kernels (P-K)

Efficiency ε = achieved ÷ P-H3 peak. Shapes are the model's real ones: DeltaNet `attn_qkv` 2048→8192, `attn_gate` 2048→4096, `ssm_out` 4096→2048; attention `attn_q` 2048→8192 (joint q+gate), `attn_k`/`attn_v` 2048→512, `attn_o` 4096→2048; shared expert 2048↔512; routed expert gate/up 2048→512, down 512→2048; router 2048→256; `lm_head` 2048→248 320; row counts M ∈ {2 048, 4 096, 8 192}.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-K1** | What ε does the bf16 path (dequant-to-scratch + `mma` GEMM) reach on those shapes, against cuBLASLt on the same data? | Standalone kernel + cuBLASLt reference; fraction of P-H3 bf16 peak per shape and M. | Plan assumes ε ≈ 0.4–0.6. Adopt the hand kernel if ≥ 0.85 × cuBLASLt on a shape; if ε < 0.35 anywhere, the tile design is reopened before more kernels are written. | `k1_bf16.json` | W0 |
| **P-K2** | What ε₈ does the int8 path (dequant to int8 in registers, `mma.s8`, per-sub-block fp32 epilogue) reach? | Standalone kernel on the same shapes with Q4_K and Q6_K bytes. | Plan assumes ε₈ = ε − 0.1. Mode F is adopted as default only if ≥ 1.3× faster than Mode R per layer-window **and** KL no worse (P-Q6). | `k2_int8.json` | W0 |
| **P-K3** | Is in-register dequantization of Q4_K/Q6_K memory-bound or ALU-bound, and what effective GB/s does it stream? | Kernel that streams expert bytes from VRAM and dequantizes into registers without MMA; `ncu` counters if they work on WSL2, else event timing. | Record. ALU-bound (< 60 % of VRAM bandwidth) → the epilogue fusion and the int8 path matter more; memory-bound → tile depth matters more. | `k3_dequant.json` | W0 |
| **P-K4** | How does the grouped MoE GEMM behave under real routing histograms? | Synthetic weights; per-layer expert-count histograms recorded by the engine traces (counts only); T = 4 096–8 192 tokens × 8 assignments. | ε_grouped ≥ 0.8 × the dense ε of P-K1 on the same bytes. Output: best tile per M_e bucket, tile-map overhead, load-imbalance loss. | `k4_grouped.json`, tile table | T3 |
| **P-K5** | How efficient is the transposed-dequant data-gradient kernel versus the forward kernel? | Same shapes, forward vs dgrad (shared-memory round trip + `ldmatrix.trans`). | dgrad ≥ 0.7 × forward efficiency. Below that, the fallback is a dequant-to-scratch + bf16 GEMM for dgrad only. | `k5_dgrad.json` | T3 |
| **P-K6** | Is the fp32 router kernel (FMA, no TF32, deterministic top-8) cheap and exact? | Time at T = 8 192; set and order agreement with the fp64 reference on random and real hidden states; crafted ties. | ≤ 1 % of layer time; identical top-8 sets and order to fp32 CPU (6.11). | `k6_router.json` | T4 |
| **P-K7** | Which fused attention-backward tiles at head dim 256 fit `sm_89` without register spills, and how fast are they? | Sweep (B_q, B_k, stages); `ptxas -v` spills; time vs the blocked fallback. | Adopt the best tile with 0 spills and ≥ 40 % of bf16 peak. Else the blocked fallback (≈ 3 % of step cost, ESTIMATE) stays the default. | `k7_attn_bwd.json` | T5 |
| **P-K8** | What does the DeltaNet chunk-state sweep cost (128 sequential chunks at T = 8 192, 30 layers)? | Time the sweep kernel on random data at real head/state shapes. | ESTIMATE was 5–10 µs per chunk ≈ 1–2 % of step time; pass if ≤ 2 %. | `k8_chunk_sweep.json` | T6 |
| **P-K9** | Are the 64×64 triangular inverse and chunk preparation accurate and fast enough? | Compare with the fp64 sequential reference over lengths 64–8 192. | Within the D4 tolerance of 6.11; no error growth with length. | `k9_chunk_prep.json` | T6 |
| **P-K10** | Does the chunked DeltaNet backward match the fp64 sequential backward for q, k, v, g, β? | T = 512 … 8 192; finite-difference spot checks. | rel. L2 ≤ 1e-3 (6.11) and no drift with length. | `k10_deltanet_bwd.json` | T6 |
| **P-K11** | Is the fused chunked cross-entropy exact and no slower than the lm_head GEMM it contains? | 256-position chunks over the 248 320 vocabulary, only weight > 0 rows, vs fp64. | loss rel. ≤ 1e-6, `dh` rel. L2 ≤ 1e-5, time ≤ 1.1 × the bare lm_head GEMM. | `k11_ce.json` | T4 |
| **P-K12** | Is the LoRA-in-epilogue fusion faster than separate LoRA GEMMs? | Fused vs two small GEMMs on all 310 module shapes. | Fused ≥ 1.1× faster, else keep separate (simpler). | `k12_lora_epilogue.json` | T3 |
| **P-K13** | What does deterministic gradient reduction cost versus atomics? | LoRA weight-gradient reduction at T = 8 192 in both modes. | Deterministic ≤ +3 % step time → stays the default. Otherwise `fast` mode is offered only for sweeps, never for gates. | `k13_det.json` | T9 |

### 11.4 Numerics and served-model semantics (P-Q, P-S)

P-Q probes decide precision questions; P-S probes settle conventions that the served model fixes and the trainer must copy exactly. Most P-S probes read the served model through llama.cpp and need no trainer code.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-Q1** | Does our dequantizer reproduce ggml bit-for-bit for every tensor type in the file? | Standalone dequant kernel vs ggml `dequantize_row_*` over **every** tensor of the served GGUF. | 100 % bit-exact (Q4_K, Q6_K, Q8_0, F32). | `q1_dequant.json` (per tensor) | W0 |
| **P-Q2** | Is our `q8_1` activation quantizer bit-identical to ggml's, including the 128-element `mmq` layout? | Compare codes, `d`, `s` on random and real activations. | 100 % identical. | `q2_q81.json` | T3 |
| **P-Q3** | **Does the learned delta survive the merge?** For each core class, how much of ΔW remains after merging into Q4_K, Q6_K, Q8_0 (same type vs raised type)? | Take the v0.2 adapter's ΔW (path to be confirmed by the author, read-only; if absent, a random low-rank Δ of matched norm, labelled weaker evidence). Form W_eff = deq(W) + ΔW; quantize; measure retention cosine and ‖Q(W_eff) − W_eff‖/‖ΔW‖; then teacher-forced KL on 64 documents via llama.cpp for each assignment. | Per class, the smallest type inside the byte budget (D22, P-E4) with retention cosine ≥ 0.98 **and** KL ≤ 1.0 × float spread (P-Q4) is chosen; Q8_0 is considered only if the engine supports it (P-E5) and the budget has room. Same-type merge is expected to fail (erasure of deltas < step/2) — the probe shows how badly, and which classes need Mode Q (D23). | `q3_merge_survival.json` (per class) | W0 |
| **P-Q4** | **What is llama.cpp's own float spread?** KL between runs that differ only by legal float differences. | `llama-perplexity --kl-divergence`, **the engine's protocol** (16 chunks of 1,024 tokens, reference = stock at batch 8): stock at batch 16 and 1,024; CPU vs GPU; `-ub` sizes; MMQ vs cuBLAS; CUDA graphs on/off; flash-attn on/off; repeat of one config. | Must reproduce the engine's MEASURED stock noise floor, mean KL **0.017353 (batch 16) and 0.018286 (batch 1,024)**; until it does, those two numbers are σ_float. **Every KL gate in this document is 1.0 × σ_float.** (These numbers compare the *same file* across implementations; a fine-tuned file is never compared with the base model's logits.) | `q4_float_spread.json` | W0 |
| **P-Q5** | Does a bf16 residual / boundary store hurt versus fp32? | Trainer forward with each, KL vs llama.cpp on 64 documents; loss-curve comparison on a 100-step run. | bf16 accepted only if KL(bf16) − KL(fp32) ≤ 0.1 × σ_float and the curves overlay. | `q5_boundary.json` | T8 |
| **P-Q6** | Mode R (bf16 reference) vs Mode F (serve-faithful int8): which is closer to llama.cpp, and by how much? | KL(trainer forward ‖ llama.cpp logits) on 64 held-out documents; per-layer error profile. | Prefer the lower KL; both ≤ 1.0 × σ_float is the goal. Mode F is default only with P-K2's speed condition. | `q6_modes.json` | T8 |
| **P-Q7** | Which internal precision does the served attention use, and what is the cheapest sufficient precision for our attention scores/PV and DeltaNet chunk GEMM fragments? | Read the CUDA FA kernel source; compare variants (fp16/bf16 PV, fp32 softmax; tf32/bf16/fp32 chunk GEMM) against per-layer llama.cpp outputs and against the fp32 sequential DeltaNet. | Pick the cheapest variant whose per-layer error ≤ 2 × the llama.cpp fused-vs-unfused difference (P-S7). | `q7_internal.json` | T5 |
| **P-Q8** | How does accumulation order and length affect dgrad/wgrad error for K up to 8 192? | fp32 block accumulation vs sequential vs fp64, bf16 inputs. | Within the 6.11 tolerances for all K. | `q8_accum.json` | T3 |
| **P-Q9** | Does cross-entropy gradient filtering (ε = 2⁻¹²) preserve the gradient? | Gradient cosine and speed, on/off, on real hidden states. | Adopt only if cosine ≥ 0.9999 **and** ≥ 5 % faster. Default off. | `q9_filter.json` | T4 |
| **P-Q10** | How does FP8 (e4m3/e5m2) behave for activations/gradients on this card? | Peak rate (with P-H3) and gradient cosine for dgrad with FP8 activations. | Measured only; not planned (Ada FP8 is 2× bf16, not 4×). Adopted only for a pass at cosine ≥ 0.999 and ≥ 1.3× speed. | `q10_fp8.json` | W0 |
| **P-Q11** | Is an int8 data-gradient path accurate enough? | Cosine of int8 dgrad vs bf16 dgrad on real tensors at several layers. | Adopt only if cosine ≥ 0.9995 at every tested layer; default bf16. | `q11_int8_dgrad.json` | T3 |
| **P-Q12** | How often does our fp32 router choose a different top-8 set than llama.cpp? | Feed llama.cpp's own per-layer hidden states (P-S1 dumps) to our router kernel; compare sets and order; also compare llama.cpp CPU vs GPU on the same dumps. | Our disagreement ≤ llama.cpp's own CPU-vs-GPU disagreement. | `q12_router_sets.json` | T4 |
| **P-S1** | Can we dump the served model's per-layer tensors by name for parity fixtures, and which conv tap multiplies the current token? | A small libllama program with a ggml eval callback (or `eval-callback`) saving `l_out-N` and block outputs for three fixed sequences; fp64 reference of the conv to settle tap order. | Dumps reproducible bit-for-bit across two runs; tap order identified. | `s1_dumps/` (hash manifest) | W0 |
| **P-S2** | Is the norm convention exactly "+1 folded in the GGUF, except `ssm_norm`"? | Compare `RMSNorm(x)·w′` with `RMSNorm(x)·(1+w)` on dumped activations for every norm type. | Match to ≤ 1e-6 relative. | `s2_norm.json` | W0 |
| **P-S3** | Is the V-head order in the GGUF the tiled order, and what inverse permutation does the oracle need? | Match the DeltaNet output with the permuted weights against the dump. | Exact match for one permutation, recorded. | `s3_vhead.json` | W0 |
| **P-S4** | Which IMRoPE convention does the served model use (adjacent pairs vs half-split; stream assignment [11,11,10,0])? | Apply both to the same Q/K and compare with llama.cpp's post-RoPE tensors. | Exactly one convention matches to ≤ 1e-6 relative. | `s4_rope.json` | W0 |
| **P-S5** | How is the MTP block wired (embedding/hidden concat order, which position predicts which token, document-boundary masking)? | Read the llama.cpp graph; dump MTP logits for a fixed sequence; test candidate wirings. | The wiring that reproduces llama.cpp's MTP logits to ≤ 1e-4. Gates D12 (MTP co-training). | `s5_mtp.json` | W0 |
| **P-S6** | How far apart are CPU-computed and GPU-computed experts (Q8_K vs q8_1 activations) in KL? | Run with all experts on the CPU, then on the GPU (`--n-cpu-moe` or `-ot`), same everything else; KL against each other and a reference. | Record. Informs whether Mode F's q8_1 choice leaves a CPU-tier residual (the engine serves ≈ 78 % of activations from the GPU bank). | `s6_cpu_gpu.json` | W0 |
| **P-S7** | Do llama.cpp's fused GDN op, its unfused chunk graph and our fp32 sequential recurrence agree? | Compare per-layer DeltaNet outputs on the P-S1 fixtures (switch via build flag or the CPU path as the unfused reference). | All three agree within the D4 tolerance; the difference defines the DeltaNet part of σ_float. | `s7_gdn.json` | W0 |
| **P-S8** | How does ggml break ties in top-k, and is softmax-over-256-then-renormalize equal to softmax-over-the-selected-8? | Tiny graph with crafted ties and logits; compare orders and weights with our kernel. | Same set, same order, weights equal to 1e-6. | `s8_topk.json` | W0 |

### 11.5 Data and tokenization (P-D)

All P-D outputs are counts, ids and pass flags. No text from the private dataset is ever printed or stored in a report.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-D1** | Does the libllama tokenizer give exactly the HF tokenizer's ids on the whole dataset? | Tokenize every document with `llama-tokenize`/libllama and with the HF tokenizer (test tooling only); compare. | 0 mismatching documents. | `d1_tokenizer.json` (counts) | W0 |
| **P-D2** | Does render → tokenize → detokenize round-trip every assistant turn, including `reasoning_content`? | libllama's Jinja engine on the GGUF template; compare to source turn by turn. | 0 failures. | `d2_template.json` (counts) | T2 |
| **P-D3** | Is a packed window equivalent to its documents run one by one? | Loss and LoRA gradients, packed vs single-document, on a 4-layer slice and on the full model. | Loss rel. diff ≤ 1e-5; gradient cosine ≥ 0.99999. | `d3_packing.json` | T9 |
| **P-D4** | What are the real average document length and the padding waste of the 64-slot alignment? | `audit-data` histogram. | Record (the 2.6 % figure is DERIVED from an inferred 1.2 k average). Flag if waste > 8 %. | `d4_lengths.json` | T2 |
| **P-D5** | Are the label masks exactly "everything after the generation prompt through `<\|im_end\|>`"? | Re-derive spans independently from marker tokens; compare token by token. | 0 mismatching tokens. | `d5_masks.json` (counts) | T2 |
| **P-D6** | Are there duplicates within train, or between train and eval? | Normalized-text hashes. | train ∩ eval = ∅; within-train duplicates reported with counts only. | `d6_dups.json` | T2 |

### 11.6 Trainer behaviour (P-T)

These check that the trainer trains correctly and that scheduling does not change results.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-T1** | Are all analytic gradients correct? | fp64 finite differences on a tiny config containing every op type (DeltaNet, attention, MoE, shared expert, LoRA, loss). | Directional-derivative relative error ≤ 1e-8. | `t1_gradcheck.json` | T7 |
| **P-T2** | Do LoRA gradients match an independent autograd model? | PyTorch fp32 oracle (test-only) on a 4-layer slice of the real model with the inverse permutations. | Cosine ≥ 0.99999 per module. | `t2_oracle.json` | T9 |
| **P-T3** | Can the trainer overfit a small set in both forward modes? | 16 documents, 200 steps, Mode R and Mode F. | Loss falls below a bar fixed beforehand (set from the first run's tokens-per-document, recorded before the run). | `t3_overfit.json` | T11 |
| **P-T4** | Are two seeded runs bitwise equal, with zero `cudaMalloc` in the step loop? | Run twice; hash LoRA state each step; allocator trace. | All hashes equal; zero allocations. | `t4_det.json` | T11 |
| **P-T5** | Is the loss independent of how the step is split into micro-batches (step-wide normalization)? | Same documents with K = 1 and K = 4 windows. | Loss and gradient rel. diff ≤ 1e-6. (v0.2 over-weighted short samples about 87×, DERIVED.) | `t5_norm.json` | T9 |
| **P-T6** | Do trainable shared-expert gates and norms help? | Ablation T-A5: off / gates / gates + norms. | Adopt only if eval loss improves by more than run-to-run noise (P-T4 gives it). | `t6_gates.json` | T14 |
| **P-T7** | Does the streamer hide the expert fetches, and are the fetch counts as planned? | Count fetches (79 per step, 35.8 GiB) and PCIe-busy and compute-stalled fractions per step at 4.9 k / 16 k / 32 k tokens. | Stall ≤ 5 % of step time at ≥ 16 k tokens; fetch count exact. | `t7_stream.json` | T10 |
| **P-T8** | Does MTP co-training protect or improve speculative acceptance in the engine? | Engine MTP acceptance rate before/after training, λ ∈ {0, 0.1, 0.3}. | Adopt the λ with the best acceptance whose CE cost is within noise. (Engine prior: decode 51.5–52.9 tok/s with MTP vs 45.4 base.) | `t8_mtp.json` | T14 |
| **P-T9** | Does the new trainer, on v0.2's recipe, learn at least as well as v0.2? | Eval-loss curve vs v0.2's ledger; **both final models scored by the same teacher-forced loss through llama.cpp** so train-serve differences cannot flatter either. | New ≤ v0.2 at equal tokens, or the difference is explained by P-X2. | `t9_parity.json` | T11 |
| **P-T10** | What exactly is "tokens per second" in the v0.2 ledger? | Read the v0.2 code and ledger; count processed vs assistant tokens per step. | A single written definition; every later comparison uses it. | `t10_definition.md` | W0 |
| **P-T11** | Does resume reproduce an uninterrupted run bitwise? | Stop at step N, resume, compare LoRA state and metrics at N+M. | Bitwise equal. | `t11_resume.json` | T11 |
| **P-T12** | How does throughput scale with step size and window size, and where is the PCIe-hiding knee? | tok/s at 4.9 k / 8 k / 16 k / 32 k tokens, windows {2 048 … 8 192}. | Record; choose the default step size from the knee. Targets in 1.4 (600 / 1 200 / 2 000 tok/s). | `t12_scaling.json` | T13 |
| **P-T13** | How much does compacting pad slots save? | Run with and without compaction. | Adopt if ≥ 2 % faster. | `t13_compaction.json` | T13 |

### 11.7 Export and engine hand-off (P-X)

These decide whether the file is exactly what was trained, and what it costs the engine.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-X0** | Are the v0.2 GGUF's expert bytes identical to a fresh `llama-quantize Q4_K_M` (no imatrix) of the original bf16? Which tensors differ? | Per-tensor SHA-256 of both files (own tool or `llama-gguf-hash`). Needs the bf16 source path from the author; the HF token is never touched. | Report identical/different per tensor class. Decides whether the served file or a fresh quantization is the base (D9). | `x0_base_identity.json` | W0 |
| **P-X1** | Is every non-core (and, in Mode L, non-router) tensor byte-identical in the output? | Re-open the output; hash every unchanged tensor payload. | `identical: N of N`. | `x1_splice.json` | T12 |
| **P-X2** | What is the export KL: merged GGUF vs the trainer's own reference logits? | Top-256 + lumped-tail reference logits from `eval`; candidate loaded through libllama in `verify`. | KL ≤ 1.0 × σ_float (P-Q4); top-1 agreement and mean \|Δ log-prob\| reported. | `x2_export_kl.json` | T12 |
| **P-X3** | Does the adapter GGUF, served by stock llama-server, reproduce the trainer, and what does it cost at decode? | `--lora` on the base; KL vs trainer; decode timing with the 310 modules × 2 extra matmuls per token (ESTIMATE +6–12 ms/token). | KL ≤ 1.0 × σ_float; decode cost recorded (the adapter is a reference, not the production default). | `x3_adapter.json` | T12 |
| **P-X4** | Is the merged GGUF a valid file for every consumer? | Parse with gguf-py; load in llama-server and in the pelias-engine tier build; 20-prompt smoke. | Loads and generates in both; metadata keys intact; `pelias.*` keys ignored. | `x4_validity.json` | T12 |
| **P-X5** | What does the merged core cost the engine in speed and hit rate? | Engine reference cell (7.0 GiB bank) **and the 7.5 GiB bank**, bypass 32, 3 launches per side, merged vs base; peak VRAM recorded; WSL2 and Windows-native. | Regression ≤ 2 % (ESTIMATE) and peak VRAM ≤ 10.7 GiB at the 7.5 GiB bank. Else the byte-budget solver is re-run with a smaller Δ_max (9.4), Mode Q is used for the classes affected, or Mode L (7.12) buys hit rate back. | `x5_engine.json` | T12 |

### 11.8 Router locality — Mode L (P-L)

Defined in 7.12. P-L1 and P-L2 need no trainer code and say how large the prize is, and what an untrained clamp costs on this exact model, before any Mode L code is written.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-L1** | What does the served model's routing look like: per-layer usage, adjacent-token reuse, static-bank hit curve (2–8.5 GiB), LRU/LFU/Belady, by prompt class and by prefill vs decode? | The engine's existing trace tooling on a held-out corpus (public-domain text + the gauntlet prompts; contents never printed). | Record, including per-layer hit, the miss-count histogram and the all-8-inside coverage curve (P-E1 has the first result). The gap between static and Belady at the reference bank bounds what any router change can win; if static is already within 0.03 of Belady, Mode L is closed. | `l1_profile.json` | W0 |
| **P-L2** | What does an untrained inference-time clamp cost on this model? | Few-line llama.cpp patch adding a per-layer [256] bias to the router logits before top-k (bias β on the resident set from P-L1, β ∈ {0.25, 0.5, 1, 2}); `llama-perplexity` and KL on 64 documents. The patch stays a loose patch. | Record the Pareto (Δhit vs ΔCE). Reproduces (or not) 2608.18261's ≈ 30 % misses at ≤ +2 % on this model; it is the cost curve training must beat. | `l2_clamp.json` | W0 |
| **P-L3** | Does the training surrogate (probability mass on R_l) track the true top-8 hit fraction? | Log both over a 200-step Mode L run. | Spearman ρ ≥ 0.9 per layer on average; else change the surrogate (7.12 item 4). | `l3_surrogate.json` | T14 |
| **P-L4** | Which (λ_res, λ_trust, lr_router) passes G-L1 and G-L2? | Reduced factorial from C0: λ_res ∈ {0 (CE-only control), 0.01, 0.02, 0.05, 0.1}; λ_trust ∈ {0.1, 0.45, 1.0}; lr_router ∈ {1e-5, 2e-5, 5e-5}. | Largest λ_res with CE ≤ 1.01 × C0 and hit ≥ C0 + 0.03. None at two router learning rates → Mode L closed. | `l4_pareto.json` | T14 |
| **P-L5** | Does training absorb the clamp (the stacking effect)? | Apply the P-L2 clamp to C0 and to the best candidate; compare ΔCE. | Candidate's ΔCE ≤ 0.5 × C0's (the paper saw 0.9 % vs 7.7 %). | `l5_stacking.json` | T14 |
| **P-L6** | Does the gain hold on domains not in the training set? | Hit rate per class (agent, code, persona, reasoning) on unseen prompts, candidate vs C0. | G-L3: each class ≥ C0 − 0.01. | `l6_breadth.json` | T14 |
| **P-L7** | Does the engine get faster with the Mode L candidate? | Re-profile the hot set from the candidate's tape; engine reference cell, 3 launches per side. | G-L4: decode above C0's by more than run-to-run noise. | `l7_engine.json` | T14 |
| **P-L8** | Is the resident set stable under Mode L, and how many rounds are needed? | Jaccard(R_l before, after) per layer. | Mean ≥ 0.9 within 2 rounds. | `l8_stability.json` | T14 |
| **P-L9** | Does llama.cpp apply a LoRA adapter to the router tensor (`ffn_gate_inp`) of this architecture? | Read the qwen35moe graph's use of `build_lora_mm` for the router; load a tiny test adapter. | Yes → form B (router LoRA) is available. No → form A (full F32 router rewrite) only. | `l9_router_lora.json` | W0 |
| **P-L10** | Does the candidate pass the quality gate? | Held-out assistant-token CE; the author's private generation and format checks (pass flags only). | G-L1: CE ≤ 1.01 × C0 and no regression on the author's checks. | `l10_quality.json` | T14 |

### 11.9 Engine co-design measurables (P-E)

Defined by the audit of Chapter 16: data the trainer needs from, or must agree with, the engine. Chapter 16.6 lists, for each, how the result is used and where it applies in this document. Owners: "trainer" probes run in this project; "engine" probes are the engine's own measurements, consumed here.

| ID | Question | Method | Pass / fail | Output | When |
|---|---|---|---|---|---|
| **P-E1** | Do the 10 full-attention layers route differently from the 30 DeltaNet layers, and what does making them 100 % resident cost? | The engine's own trace simulator (`waterfill_sim`, `export_hot_set` logic) on its recorded routing (121,160 decode tokens; 70/30 fit/held-out split), read-only: per-layer hit, zero-miss share, all-8-inside coverage by set size, and a constrained bank with the full-attention layers fully resident. | **Done once (2026-10-23)** — baseline reproduced exactly (held-out hit 0.8273, any-miss 0.639, 4,170 experts); full-attention layers hit 0.8299 vs 0.8265 elsewhere; 100 % residency of those layers uses 4.605 of 7.5 GiB and drops overall hit to 0.7114 and the others to 0.616. Re-run on every Mode L candidate; layer weights stay uniform unless P-E2 says otherwise. | `e1_layer_type.json`, script spec in 16.6 | W0 |
| **P-E2** | What does one layer cost as a function of its miss count (0…8), layer type and context depth? | Engine E0 per-layer timers (`e0_timeline.patch`); decode at context short / 4 k / 12 k / 32 k / 131 k; regress layer latency on miss count, with layer type as a factor. | Record the cost of the first miss (the CPU trip) c₀ and of each extra miss c₁ per layer type. If c₀ ≫ c₁ the zero-miss share, not the mean hit rate, is the objective (7.12); if full-attention layers cost more per miss, their loss weights rise in proportion. | `e2_miss_cost.json` | W0 (engine) |
| **P-E3** | Is zero-spill on chosen layers feasible for this router? What are the logit margins between the in-bank top-8 and the best out-of-bank expert? | Router logits per layer (`ffn_moe_logits-N` via the P-S1 callback) over the trace corpus; for candidate sets R_l of 96 … 224 experts compute the gap between the 8th-best in-set logit and the best out-of-set logit; also record the gap between the 8th and 9th best logit overall (near-tie flips, 16.5 B5) and compare it with the logit difference between llama.cpp's CPU and CUDA paths on the same tokens (P-Q4). | Feasibility = share of tokens with a positive gap (this is the base zero-miss rate; P-E1 gives 0.36 at the engine's sets). The hinge margin m is set to the 25th percentile of positive gaps (DECISION rule). If the median needed shift exceeds the logit range the router can move under the trust region (P-L4), the zero-spill mode is closed. | `e3_margins.json` | W0 |
| **P-E4** | How much VRAM does the engine really leave for the core, per bank size and context? | Engine reference cell at bank 7.0 / 7.5 GiB, prefill micro-batch default / 2048 / 4096, and context 12 k / 32 k / 131 k with core variants built only to measure bytes (`llama-quantize --tensor-type` overrides: served, all-Q6_K, mixed, all-Q8_0); peak VRAM from NVML. | Replaces the DERIVED Δ_max = 0.16 GiB (default micro-batch: ceiling 10.7 − peak 10.44 − 0.1 margin) and none (`-ub 2048`: peak 10.69, −0.09), on the whole-card basis; also records NVML used before the engine starts, so the process-only reading (16.3.1) can be computed. Sets the solver input of 9.4 for the micro-batch the engine ships with. | `e4_vram_budget.json` | W0 |
| **P-E5** | Which weight types will the engine's spine and expert kernels support? | Read the engine blueprint (E4/E5/E7) and ask the author; record as a matrix {type × spine/expert/CPU/GPU}. | A written DECISION. Today: experts `Q4_K`/`Q6_K`; spine = upstream kernels as reference; KV `q8_0`. Output files may use only supported types (D22). | `e5_type_matrix.md` | W0 |
| **P-E6** | Is our port of ggml's reference quantizers bit-exact for Q4_K, Q6_K and Q8_0? | CPU port of `quantize_row_*_ref` (MIT) vs ggml, on random blocks with outliers, on every core tensor, and on merged `W_eff`; then the GPU port vs the CPU port. | 100 % bit-identical (codes, `d`, `dmin`, packed scales). Gate for Mode Q and for the exporter. | `e6_quantizer.json` | W0 |
| **P-E7** | Do the served base weights sit exactly on their grid? (re-quantization idempotence) | For each core tensor: fraction of blocks where `Q_T(deq(block)) == block`, per type and class (uses the P-E6 CPU port). | Record. ≈ 100 % confirms the erasure model of 4.3 (same-type merge erases deltas < step/2); well below 100 % means partial survival and changes the P-Q3 expectations. | `e7_idempotence.json` | W0 |
| **P-E8** | Does Mode Q (training through the exact quantizer, 4.10) close the merge gap, and what does it cost? | 4-layer slice first, then the full model: STE-through-`Q_T` fine-tune vs LoRA-then-merge at Q6_K; held-out CE through llama.cpp after export; step overhead. | Adopt for a class if its post-export CE is lower than the type-raise alternative inside the budget, or equal at fewer bytes; overhead ≤ 5 % of step time. | `e8_modeq.json` | T14 |
| **P-E9** | What is the alignment of the served file's expert tensors, and of rows and blocks? | Header-only read (gguf-py): absolute offset mod 16 B / 4 KiB / 2 MiB for all 123 expert tensors; bytes per row and per block for each type. | **Done once (2026-10-23, READ-SRC):** all 123 offsets are multiples of 16 B; 9 are 4 KiB-aligned; none is 2 MiB-aligned; data starts at byte 10,991,104; `general.alignment` absent (32). Q4_K rows are 1,152 B (a 16-B multiple); **Q6_K rows are 420 B (not a 16-B multiple; 420 mod 16 = 4)**. Expert tensors are 144 or 210 MiB (2 MiB multiples). Re-run on every export (the layout manifest, 16.5 B4); the expert-last layout of B4 is checked here too (llama.cpp must load it with logits identical to the natural layout). | `e9_alignment.json` | W0 (done) / T12 |
| **P-E10** | Does a 16-byte-load, shared-memory-staged kernel lift zero-copy compute reads from 9.59 toward 26.2 GB/s, and does Q6_K need row padding? | Engine `wide_load.cu` (E0/E5): same pinned expert bytes read by byte loads, `uint4` loads, `uint4` + shared-memory staging; Q6_K rows raw (420 B) vs padded (432 B). | Engine-owned. Pass: ≥ 20 GB/s. The result decides whether the export manifest must demand padded Q6_K rows (+2.9 % bytes on 21 of 41 down tensors) or the engine repacks. **Until measured, the byte-load explanation is the engine's own *unverified* hypothesis.** The checksum kernel (26.2 GB/s) and the compute kernel (9.59 GB/s) read the *same* pinned pages, so page granularity alone cannot explain the gap (DERIVED); load width and access pattern can. Q6_K blocks are 210 B (210 mod 16 = 2), so only the stream, not each block, can be 16-B aligned: the kernel must stage 16-B-aligned chunks and parse blocks from shared memory; the row padding above only fixes row starts. | `e10_wide_load.json` | W0 (engine) |
| **P-E11** | Do 2 MiB pages change zero-copy read speed (WSL2 THP and Windows-native `MEM_LARGE_PAGES`)? | Engine P-H5-style read test with 4 KiB vs 2 MiB-backed pinned buffers, on WSL2 and Windows-native. | WSL2 here is already booted with `transparent_hugepage=always` (`.wslconfig`, machine state), and the engine measured 44.86 GB/s with 2 MiB pages against 45.81 GB/s with 4 KiB pages for the CPU read test (MEASURED), i.e. no gain. If the GPU path shows ≥ 5 %, the manifest flag `page_aligned` becomes required and the exporter pads expert tensors to 2 MiB (cost computed from P-E9). Until then huge-page alignment is optional. | `e11_pages.json` | W0 (engine) |
| **P-E12** | Does the CPU-vs-GPU expert-output difference grow with activation outliers? | From P-S1 dumps: expert outputs via llama.cpp's CPU kernel (Q8_K activations, 256-blocks) and CUDA kernel (q8_1, 32-blocks) for sampled (token, layer, expert) triples; relative difference vs the outlier index ρ = max\|x\| / rms(x) of the input `u` and of the SwiGLU output `h`, per block. | Mode B / activation-outlier regulariser (D24) adopted only if Pearson r ≥ 0.3 **and** the top 5 % of ρ account for ≥ 50 % of the excess KL (P-S6). | `e12_outliers.json` | W0 |
| **P-E13** | Which accumulator types and reduction trees do the engine's kernels (when written) and llama.cpp's use? | Static audit of `expert_q4k*.{cu,cpp}`, `expert_q6k*.{cu,cpp}` and ggml `mmvq`/`vec_dot`; test with injected outliers against fp64. | Rule recorded for the engine: integer dot products accumulate in int32, scales apply in fp32, never fp16. Violation = engine bug, not something the trainer compensates. | `e13_accumulators.md` | T12 (engine) |
| **P-E14** | Does the trainer's hot-set writer reproduce the engine's file byte for byte? | Same traces and budget through `export_hot_set.py` and the C++ writer; compare text and JSON provenance; run the engine's parser mirror `parse_text`. | Byte-identical text; parser accepts; zero-slot budget accounting equal. | `e14_hotset_parity.json` | T12 |

### 11.10 Wave 0 — suggested order, and what each result changes

| Step | Probes | Why this order | Decision it feeds |
|---|---|---|---|
| 1 Platform baseline | P-H8, P-H1, P-H2, P-H3, P-H4, P-H10 | Everything else is normalised by these; the soak tells whether the overclock can be trusted for a multi-hour run | ε denominators; run clocks; canary policy (15.3) |
| 2 Transfers and memory | P-H5, P-H6, P-H7, P-M3, P-M4, P-H9, P-H12, P-H13, P-H11 | They set the PCIe-hiding knee, the host memory plan and whether graphs and copy overlap exist | Streamer design (5.4, 5.5); fallbacks of 5.3; platform default |
| 3 Kernel efficiency | P-K1, P-K2, P-K3, P-Q10 | The speed story rests on ε; if ε < 0.35 the tile design is reopened **before** more kernels are written | Mode R vs F; targets of 1.4 |
| 4 Served truth | P-Q4 first, then P-S1, P-S2, P-S3, P-S4, P-S7, P-S8, P-S6, P-S5, P-Q1 | P-Q4 sets every KL gate; P-S1 produces the fixtures the others reuse | KL gates; conventions; MTP wiring; dequant parity |
| 5 Data, base and engine facts | P-D1, P-T10, P-X0, P-Q3, P-E4, P-E5, P-E6, P-E7, P-E3, P-E12, P-L1, P-L2, P-L9 (P-E1, P-E9 done) | P-Q3 and P-E4/E5 decide the core's types; P-E6 gates Mode Q; P-L1/L2/E3 decide whether Mode L-Z is worth building | Core type assignment (9.4); base file (D9); Mode L go/no-go; Mode B/N go/no-go |
| 6 Engine-owned | P-E2, P-E10, P-E11 | Measured by the engine project; consumed here | Loss weights (7.12); export layout (9.3, 16.5 B4) |

Wave 0 ends with a one-page summary that restates each design switch with its measured value; stage T0's exit gate (Chapter 12) is that the author has read it and signed off the switches.

### 11.11 Result file schema

```json
{
  "probe": "P-Q4",
  "date": "2026-10-24T09:12:00-04:00",
  "machine_state_sha": "<hash of the recorded machine-state file>",
  "label": "MEASURED",
  "inputs": { "gguf_sha256": "...", "llama_cpp": "2ca15f540", "args": ["..."] },
  "values": { "...": 0.0 },
  "thresholds": { "...": 0.0 },
  "pass": true,
  "raw": ["relative/path/to/raw.log"],
  "notes": "free text; no dataset content"
}
```

The ledger line appended for each run is `date · probe · pass/fail · key value · raw-file hash`, in the same style as the engine's claim-to-log ledger, so a number in this document can always be traced to a file.

### 11.12 Wave 0 results so far — the hardware set (2026-10-23/24)

The hardware probes of 11.2 exist as one standalone program (C++/CUDA; one JSON per probe in the schema of 11.11; it never overwrites a result). It was run in full on **WSL2** (2026-10-23, 68 minutes) and on **Windows-native** (2026-10-23/24, 68 minutes; built with nvcc 13.4 and Visual Studio 18 Professional), with the overclock on and nothing else using the GPU. Every step exited 0. P-M4 has not run (it needs the model path). The raw files are kept in the author's backup folder, not in this repository. Everything below is MEASURED unless marked; "A / B" means WSL2 / Windows-native.

| Probe | Result (A / B) | What it changes |
|---|---|---|
| P-H8 free VRAM | Total 11.99 GiB. After the CUDA context exists: CUDA free **10.83 / 10.85** GiB, NVML free 10.95 / 10.85. Context cost 154 / 138 MiB. NVML used before the context 914 / 1,031 MiB (held by Windows). BAR1 total 16,384 MiB; used 0 / 16,355 MiB. The same 10.83 is in 21 engine logs (a fixed driver budget; inference) | The ceiling (10.7, D26); only 0.13 / 0.15 GiB of physical room above it |
| P-H1 device | 46 SMs; opt-in shared memory 101,376 B (99 KiB) per block; the driver reports 5 async copy engines | Tile constants as planned |
| P-H2 bandwidth | VRAM read 523 / 534, write 448 / 454, copy kernel 473 / 481, `cudaMemcpy` device-to-device 495 / 504 GB/s (the last two count read + write). Host DRAM read, 1 thread 37.1 / 37.9, best at 4 threads 43.7 / 44.8 GB/s | Roofline inputs; VRAM read is above the 504 GB/s stock spec (the memory is overclocked) |
| P-H3 tensor rates (60 s each, sustained) | bf16 65.2, fp16 64.9, int8 259.5, fp8-e4m3 129.8 T(FL)OPS, identical on both platforms to 0.1; 99.8 % of the burst rate; SM clock 2,760–2,775 MHz; 0.511 / 0.511 / 2.044 / 1.022 per SM per GHz | Replaces the DERIVED overclock peaks 65.4 / 261 (−0.3 % / −0.7 %). int8 is 3.98× and fp8 1.99× bf16: the 4× assumption holds. Stock-clock values are not measured |
| P-H10 canary | 10,000 GEMM runs, 0 mismatches, 1 distinct hash, CPU-check error ≤ 1 % of tolerance (both platforms) | R3: no bit flip seen at the overclock in this run |
| P-H4 soak (30 min) | 17,685 / 17,700 cycles, 0 mismatches. SM clock median 2,760 (per-minute means 2,760.0–2,760.1); 43–55 / 49–56 °C (median 52 / 53); power median ≈ 107 W of the 200 W limit. Memory clock under load: 11,251–11,501 MHz on A, a flat 11,501 on B. The driver's throttle-reason field reads "gpu_idle" (raw mask 1) on every sample on both platforms while the GPU is flat out | The throttle-reason field is unusable; the stop rule of 15.3/15.8 uses clocks and temperature |
| P-H5 transfers (256 MiB) | Pinned H2D 26.3 / 26.3 and D2H 26.3 / 26.3 GB/s; pageable H2D 10.3 / 11.8, D2H 10.8 / 11.8; zero-copy kernel read 26.3; three streams together 26.1 / 26.2. Small copies (median): 4 KiB 73 / 31 µs, 1 MiB 113 / 72 µs | 5.5 uses 26.1 (about 1 % low). The 20 GB/s pass line is met |
| P-H6 contention | H2D 26.1 alone; with 4 / 8 / 16 CPU reader threads 26.0 / 25.3 / 23.6 on A (worst ratio 0.90; B worst 0.87); combined CPU + H2D 46.6–48.7 GB/s on A (engine prior 47.6) | Roofline slowdown factors |
| P-H7 pinned limit | **32 GiB** (the largest size tried; 34.4 GB) held on both: 0 MiB swapped in, DMA round trips pass. Allocation 0.76–1.0 s per GiB on A, 0.069 s per GiB on B. Host available memory falls 1.01 GiB per pinned GiB (A 45.9 → 13.7; B 55.6 → 23.4). The real limit is not found | R7 closed. The 28–33 GiB plan of 5.3 fits |
| P-M3 host hold | 24 GiB pinned + 4 GiB pageable for 30 min: 0 mismatches, 0 MiB swapped in, H2D during the hold median 25.8 / 26.0 GB/s. 10-ms-sleep overshoot: A median 0.082, p99 0.375, max 1.70 ms; B median 6.14, p99 6.48, max 16.9 ms; no stall over 50 ms on either | The B figure is probably the default 15.6 ms Windows timer tick, not a stall (INFERENCE; the Windows build asks for no finer timer), so the two are not comparable yet. No jitter threshold exists until an idle run does |
| P-H9 graphs and flush cost | Eager launch 4.70 / 2.18 µs per kernel, graph replay 0.716 / 0.707 µs (85 % / 68 % saving; A's eager figure was 2.25 µs in the short pass, so it depends on run length); counters exact; parameter update works; 10⁵ replays. `cudaStreamQuery` flush: idle stream 0.098 / 0.091 µs; busy stream median 76.6 / 8.05 µs per call; empty launch + flush, pipelined, 48.6 / 24.2 µs against 5.67 / 2.89 µs for the launch alone | Graphs adopted (pass: ≥ 30 %). The flush is host time, hidden while the GPU has queued work (5.4) |
| P-H12 overlap | Five copy engines. Ordinary submission: the work runs one after another on the GPU timeline (all-three efficiency 0.47 / 0.52 for many short tensor launches, 0.00 for one long launch, 0.39 / 0.42 for the VRAM-read kernel). With a flush after each enqueue, compute first, interleaved copies: **0.92 on both**; H2D + D2H pair 0.84 | The flush rule (D27, 5.4); `S_w` and `S_x` overlap at 0.84, not 1.0 |
| P-H13 display timeout | Stages of 2, 5 and 30 s (one kernel per SM, about 60 W), a 10 s queued set and a 10 s graph: all completed, device usable afterwards, no errors. B: no display-driver events; the author saw nothing on screen on either path. The device attribute "kernel exec timeout" reads 1 and the Windows timeout is at its default (2 s) | Not a constraint on this machine for ≤ 30 s of this load. Heavier kernels and other display setups are unmeasured |

**What these results change in the design.**

1. **The ceiling is 10.7 GiB** (D26). The card has 11.99 GiB, but CUDA reports only 10.83 / 10.85 GiB free once a context exists, identically on every run and in 21 engine logs; the first P-H8 rule (free ≥ ceiling + 0.3 = 11.2) could never pass. What the number is measured against is open (16.3 X15; 16.7 decision 7).
2. **The streamer must flush** (D27): on both platforms copy/compute overlap exists only with a `cudaStreamQuery` after each enqueue (5.4, R6).
3. **R7 is closed**: 32 GiB pinned was held for 30 minutes on both platforms and the limit is above what was tried.
4. **The peak-rate inputs of the performance model are measured** (13.2). The tables of 13.3 were computed with 65.4 / 261 and move by at most 0.7 %.
5. **The two platforms agree on what the GPU does** (P-H2, P-H3, P-H5, P-H12). They differ in host-side costs: the flush (77 vs 8 µs), pinned allocation time (0.8–1.0 vs 0.07 s per GiB), small-copy latency (73 vs 31 µs for 4 KiB) and the memory clock under load (11,251–11,501 vs 11,501 MHz; cause not examined). That is the P-H11 data for P-H3, P-H5 and P-H9. It says nothing about the engine's +25–28 % decode difference, which needs the engine's own runs.

**Still unmeasured, each with the way to measure it.**

* P-M4 (model-file read speed): needs the model path (`PELIAS_GGUF` or `-Gguf`); the path is never written to a result.
* Stock-clock values of P-H3, P-H10 and P-H4: reset the overclock tool, rerun with the tag `stock`.
* The Windows jitter figure: ask for a finer timer in the Windows build (`timeBeginPeriod(1)` or a high-resolution waitable timer), rerun P-M3; an idle run (`--pinned-gib 0 --pageable-gib 0`) gives the comparison for a threshold.
* The pinned limit above 32 GiB: raise `--max-gib`.
* The memory clock under load (11,251 vs 11,501 MHz, and 11,501 at idle): `nvidia-smi dmon` across idle → load on each platform.
* Resizable BAR: no on/off comparison exists. Switch it off in the BIOS (BAR1 should then read 256 MiB), rerun P-H2, P-H5, P-H6 and P-H12 with the tag `rebar_off`, and switch it back on.
* The display timeout for full-power tensor kernels, and for a screen on the same GPU in laptop display modes.
* VRAM drift while the machine is in use (the disturbance test). No run logged GPU memory over time: the 30-minute soak's sampler records clocks, temperature and power only, so an app starting mid-run was never seen. What exists: the memory Windows held before our context was 914 and 1,070 MiB on the two WSL runs and 1,015 and 1,031 MiB on the two Windows runs, while CUDA free stayed 10.830 / 10.846 GiB. Way to measure: add NVML used and CUDA free (peak and floor) to the soak's sampler, then during a soak open a browser, play a video and start a game and record the largest move; the margin is that drift plus a small buffer. This needs a change to the probe source, which has not been made.
* Whether memory is held between probes. Before 2026-10-24 only P-H8 logged memory before its context. The run scripts now write `vram_log.tsv`, so the next runs show it; if a probe fails, `RETRY_ON_FAIL=1` shows whether flushing matters.
* Hash time for the model key (3.3, U4): time a SHA-256 over the model file, next to the read rate of P-M4.
* The power plan (Balanced during every run): rerun on High performance.
* Whether the engine's own copy/compute overlap improves with the flush: engine side (held); add `cudaStreamQuery` after its enqueues and compare the token rate.
* Native Linux: not measured (tabled).

---

## 12. Order of operations — stages T0 to T14

The order follows one rule: **build the most uncertain thing first, in the cheapest form that can be tested against something trusted.** That puts measurement (Wave 0) before design commitments, the four independent kernel families before the integration, the forward pass before the backward pass, and optional modes last. The proposals' 8-phase order (2.9) is extended to fifteen stages; each has a deliverable, an exit gate, the probes it runs, and a rollback. Relative size labels (S/M/L/XL) are ESTIMATES of effort, not durations.

### 12.1 Overview

| Stage | Name | Needs | Deliverable | Exit gate (all must pass) | Probes run | Size |
|---|---|---|---|---|---|---|
| **T0** | Measure and scaffold | — | Repo tree, CMake, probe programs, Wave 0 results, the one-page switch summary (11.10) | The author has read the summary and signed the switches (precision plan, mode defaults, Δ_max, Mode L go/no-go) | all Wave 0 | L |
| **T1** | Foundations | T0 | `hal` (device, streams, events, arenas), VRAM governor, `io` (GGUF mmap reader, tensor map), `plan` (memory solver), test harness | Lists all 753 tensors with types; a plan above 10.7 GiB is refused; clean-checkout build twice gives identical test results | P-M1 | M |
| **T2** | Data pipeline | T1 | Tokenizer and chat template through libllama, packer, masks, audits A-LEN/TOK/TPL/MASK/DUP/SPECIAL/FAMILY, `audit-data` | Zero mismatches in P-D2, P-D5; audit report with counts only | P-D2, D4, D5, D6 | M |
| **T3** | Quantized GEMM family (G) | T1 | G8 dequant, G9 `q8_1` quantizer, bf16 GEMM, int8 GEMM, grouped MoE GEMM with tile map, transposed-dequant dgrad, LoRA epilogue | Rung-1 tolerances (6.11) incl. sanitizers; efficiency recorded | P-Q2, Q8, Q11, K4, K5, K12 | XL |
| **T4** | Elementwise, router, loss | T1 | RMSNorm, SwiGLU, gated norm, L2 norm, router + counting sort + combine, chunked CE | Rung 1; router identical to fp64 on all test shapes | P-K6, K11, Q9, Q12 | M |
| **T5** | Attention (A) | T1 | IMRoPE, fused forward/backward at head-dim 256, blocked fallback | Rung 1; rotation matches llama.cpp (P-S4) | P-K7, Q7 | L |
| **T6** | DeltaNet (D) | T1 | Gates, conv1d, chunk prep, state sweep, chunked forward and backward, fp64 and fp32 sequential oracles | Rung 1; forward matches llama.cpp's fused op (P-S7); backward ≤ 1e-3 vs fp64 | P-K8, K9, K10 | XL |
| **T7** | One layer, then a tiny model | T3–T6 | Layer forward/backward for each layer type; tiny-config model with every op | Rung 2 (per-layer parity vs llama.cpp dumps); P-T1 finite-difference gradient check passes | P-M2, P-T1 | L |
| **T8** | Full forward vs served truth | T7 | 40 layers + MTP block + lm_head with simple (unoptimized) expert streaming; modes R and F | Rung 3: truncated KL ≤ 1.0 × σ_float on 64 held-out documents, per-layer profile flat | P-Q5, P-Q6 | L |
| **T9** | Backward, LoRA, optimizer | T8 | Backward sweep with the routing tape, LoRA, AdamW, step-wide normalization, deterministic reductions | Rung 4 (oracle C) and invariants: packed = unpacked, K = 1 = K = 4 | P-T2, T5, D3, K13 | L |
| **T10** | Streamer and layer-major scheduler | T9 | Three streams, expert slots, boundary store, window planner, governor under load | Zero allocations in the step loop; 79 fetches / 35.8 GiB per step; stall ≤ 5 % at ≥ 16 k tokens | P-M5, P-T7 | L |
| **T11** | Training loop, eval, checkpoints | T10 | `train`, `eval`, four registries, resume, metrics; the **v0.2-parity run** | Rung 6: overfit passes in both modes, curve parity, bitwise resume | P-T3, T4, T9, T11 | M |
| **T12** | Export, verify, engine hand-off | T11 | Adapter GGUF; merged GGUF byte-splice with the byte-budget solver and exact quantizers; hot-set writer and layout manifest (16.5); `verify` | Rungs 7–8: hash proof, KL gates, engine smoke, regression and VRAM gates | P-X1–X5, P-E9, E13, E14 | L |
| **T13** | Performance engineering | T11 | Adopt the int8 forward (if P-K2 and P-Q6 allow), fusions, CUDA graphs, tile tuning, pad compaction | Targets of 1.4 met, or the gap is explained by measurement; rungs 1–7 still pass | P-T12, T13 | L |
| **T14** | Optional modes and ablations | T12 | MTP co-training, Mode L (and L-Z), Mode Q, Mode B/N, ablations T-A1…A11 | Each mode's own pre-declared gates; losers closed with the result written down | P-T6, T8, E8, L3–L8, L10 | XL |

### 12.2 Stage detail — build, test, rollback

* **T0.** *Build:* probe programs (standalone), `probes/` runner, machine-state recorder, the tree of 5.2 with empty modules and CMake targets, `tests/ref/` skeleton. *Test:* each probe's own pass/fail. *Rollback:* none needed (no product code); if a probe fails its design switch, the switch flips and the affected stage's plan is edited before it starts.
* **T1.** *Build:* `hal/arena` (bump arenas per lifetime: weights, step, window, scratch), `hal/governor` (sums arena capacities against 10.7 GiB; `cudaMemGetInfo` cross-check), `io/gguf` (header parse, mmap, tensor map by name — Appendix A), `plan/solver` (window sizes, slots, fetch schedule). *Test:* a synthetic plan exceeding the ceiling aborts; the served GGUF round-trips its tensor table. *Rollback:* tag `t1`; nothing depends on it yet.
* **T2.** *Build:* `tok` (libllama tokenizer, template renderer), `data` (JSONL reader, `reasoning_content` handling, packer with 64-slot alignment, span masks, audits). *Test:* P-D2/D5 plus the synthetic public-domain fixtures. *Rollback:* a failed audit stops the pipeline by design; code is independent of the GPU stages.
* **T3–T6.** Each family is built to its section of Chapter 6 with its CPU/fp64 reference first, then the kernel, then the efficiency probes. *Parallel:* T3, T4, T5, T6 share only `hal`. *Rollback:* per-family tags; a family that misses its efficiency bar falls back to the simpler variant named in its section (dequant-to-scratch + bf16 GEMM for G; blocked attention for A; unfused chunk path for D).
* **T7.** *Build:* `model/layer_{deltanet,attention}` and `model/moe`, the tiny-config builder (random weights at real head dimensions, 2–4 layers). *Test:* finite differences in fp64 (P-T1); per-layer outputs vs llama.cpp dumps on three fixed sequences (rung 2). *Rollback:* tag `t7`; a failing block is bisected against oracle B.
* **T8.** *Build:* the forward-only model driver with a naive streamer (copy each layer's experts on the compute stream), lm_head with chunked CE, MTP head. *Test:* KL vs llama.cpp (P-Q6); P-Q5 boundary dtype. *Rollback:* if KL exceeds σ_float, the per-layer error profile localises the layer; fix there, never loosen the gate (loosening needs a dated waiver, 10.3).
* **T9.** *Build:* backward sweep (recompute from boundary with the tape), LoRA, optimizer, loss normalization, deterministic reductions. *Test:* LoRA gradients vs oracle C on a 4-layer slice; K-invariance (P-T5); packing equivalence (P-D3). *Rollback:* tag `t9`; gradient mismatches are localised by module.
* **T10.** *Build:* `stream` (three streams, slot ring, events), `sched` (window planner, layer-major loops), boundary store (pinned), governor integration. *Test:* P-T7, P-M5; a 30-minute run under `compute-sanitizer` at small scale. *Rollback:* the naive T8 streamer remains selectable (`--sched naive`) as the reference.
* **T11.** *Build:* `train` loop, `eval`, registries (weights, optimizer, data cursor/RNG, MTP), metrics (step-wide), resume. *Test:* P-T3/T4/T9/T11; the parity run reproduces v0.2's recipe on the author's data (private; counts and curves only). *Rollback:* none; this is the first complete trainer and is tagged `v1.0-alpha`.
* **T12.** *Build:* `export` (adapter writer, merged writer, quantizers, solver, hot-set writer, manifest), `verify` (separate process, libllama). *Test:* P-X1…X5, P-E9/E13/E14. *Rollback:* the adapter GGUF alone is a valid fallback release.
* **T13.** *Build:* tuned tiles and fusions from the Wave 0 and stage probes; CUDA graphs if P-H9 allowed; compaction if P-T13 pays. *Test:* every rung-1 test again, plus bitwise determinism between fused and unfused in deterministic mode. *Rollback:* every optimization sits behind a flag and is admitted only if it matches the slower path (6.5).
* **T14.** *Build and test:* in the order MTP (needs P-S5) → Mode Q (needs P-E6) → Mode L (needs P-L1/L2/E3) → Mode B/N (needs P-E12), each as a separate branch with its own gates (7.6, 4.10, 7.12, 16.5). *Rollback:* a mode that fails its gate is closed; the code path is deleted or left disabled, and the null result goes into the ledger.

### 12.3 Critical path and what can run in parallel

```
T0 ──► T1 ──┬─► T2 ─────────────────────────────────────────────┐
            ├─► T3 ─┐                                            │
            ├─► T4 ─┼─► T7 ─► T8 ─► T9 ─► T10 ─► T11 ─┬─► T12 ───┼─► T14
            ├─► T5 ─┤                                  └─► T13 ───┘
            └─► T6 ─┘
```

The critical path is T1 → (the slowest of T3/T6) → T7 → T8 → T9 → T10 → T11 → T12. T2 and the independent kernel families can proceed in parallel. Wave 0 can change the plan: if P-K1 shows ε < 0.35 the T3 tile design is reopened; if P-Q3 shows the core cannot be merged within the byte budget, Mode Q moves from T14 into T9–T12 (it changes the forward).

### 12.4 What "finished" means

| Question | Evidence | Where |
|---|---|---|
| Does it compute what the served model computes? | Rung 3 KL ≤ 1.0 × σ_float, both modes | T8 |
| Are the gradients right? | Rung 4 | T9 |
| Is it faster than v0.2, by how much? | P-T12 against 248 tok/s median; targets 600 / 1,200 / 2,000 | T13 |
| Is the file what was trained? | Rung 7: hash proof, KL | T12 |
| Does the engine still meet its budget with it? | Rung 8: VRAM ≤ 10.7 GiB at the 7.5 GiB bank, regression ≤ 2 % | T12 |
| Is the model better for the author? | Rung 9, ablations T-A1…A11 | T14 |

---

## 13. Performance model

### 13.1 The model

Per optimizer step with `N` processed tokens, window `T`:

```
t_step  = max( t_compute , t_stream )  +  t_serial
t_compute = N · F(T) / ( P · ε · f )            F(T) = FLOP per token (20.39 G at T = 8 192)
t_stream  = 2 · B_experts / BW_h2d              B_experts = 18.16 GiB;  2 passes (forward + recompute/backward)
```

`P` is the sustained tensor rate (bf16 or the int8/bf16 mix), `ε` the GEMM efficiency (probes P-K1/K2), `f` the share of time that is not large GEMMs (norms, recurrences, softmax, gathers, launch gaps — assumed 0.75, probes T-series refine). `t_serial` covers the optimizer step, checkpoints and host work that cannot overlap; at these step sizes it is below 1 % (ESTIMATE).

### 13.2 Inputs and their labels

| Input | Value | Label |
|---|---|---|
| Training FLOP per token | 20.39 G at T = 8 192; 19.64 G at T = 6 144 | DERIVED (6.10) |
| bf16 FP32-acc peak, stock / overclock | 58.3 (DERIVED) / **65.2 sustained** (MEASURED, P-H3) TFLOPS | stock: DERIVED from published 4090 figures × SM and clock, stock-clock repeat not run; overclock: P-H3. The tables of 13.3 were computed with 65.4 and are not re-run (difference −0.3 %) |
| int8 peak, stock / overclock | 233 (DERIVED) / **259.5 sustained** (MEASURED, P-H3) TOPS | as above; the tables of 13.3 used 261 (difference −0.7 %) |
| int8-eligible share of FLOPs | 52 % at T = 8 192 | DERIVED (6.10) |
| Expert bytes per sweep | 18.16 GiB | READ-SRC |
| PCIe host→device | 26.1 GB/s (engine); 26.3 GB/s pinned bulk, H2D and D2H | MEASURED (engine); P-H5 measured 26.3 on both platforms |
| ε, ε₈ | 0.3–0.6, ε − 0.1 | ESTIMATE; P-K1/K2 |
| f | 0.75 (0.6 as a pessimistic case) | ESTIMATE |

### 13.3 Sensitivity — training tokens per second

Window T = 8 192 (stock clocks). Cells are *bf16-only / mixed int8 forward*.

| | ε = 0.3 | ε = 0.4 | ε = 0.5 | ε = 0.6 |
|---|---|---|---|---|
| f = 0.75 | 643 / 956 | 858 / 1,318 | 1,072 / 1,675 | 1,287 / 2,031 |
| f = 0.60 | 515 / 765 | 686 / 1,054 | 858 / 1,340 | 1,029 / 1,625 |

Window T = 6 144 (a typical v0.2-parity step of ≈ 4.9 k tokens padded to one window), stock clocks:

| | ε = 0.3 | ε = 0.4 | ε = 0.5 | ε = 0.6 |
|---|---|---|---|---|
| f = 0.75 | 668 / 1,012 | 891 / 1,397 | 1,113 / 1,778 | 1,336 / 2,157 |
| f = 0.60 | 534 / 810 | 713 / 1,118 | 891 / 1,422 | 1,069 / 1,726 |

At the author's overclock (T = 8 192, f = 0.75): ε = 0.4 → 962 / 1,477; ε = 0.5 → 1,202 / 1,878; ε = 0.6 → 1,443 / 2,277 (DERIVED).

**The ceiling** (ε = 1, f = 1, no other costs) at T = 8 192 is 2,859 tok/s bf16-only or 4,708 tok/s mixed at stock clocks (3,206 / 5,279 overclocked); at T = 6 144, 2,969 / 5,014 (3,329 / 5,621). The measured result will be reported as a fraction of this ceiling, as the engine reports its own.

### 13.4 The stream bound

Expert traffic is fixed per step: 2 × 18.16 GiB = 39.0 GB at 26.1 GB/s = **1.49 s**, independent of how many tokens the step holds. It is hidden whenever compute time exceeds it:

| Compute rate | Step of 4.9 k tokens | 16 k | 32 k |
|---|---|---|---|
| 600 tok/s | 8.2 s — hidden | 27.3 s — hidden | 54.6 s — hidden |
| 1,200 tok/s | 4.1 s — hidden | 13.7 s — hidden | 27.3 s — hidden |
| 2,000 tok/s | 2.5 s — hidden | 8.2 s — hidden | 16.4 s — hidden |

So the stream is not the bottleneck at any target rate for steps of 4.9 k tokens or more; below ≈ 3 k tokens at 2,000 tok/s it would be. What *can* break this is the boundary-tensor traffic and host-DRAM contention (5.5, P-H6) and copy-engine overlap (P-H12); the table above assumes both are fine and the probes decide.

### 13.5 Time to train

For the v0.2 data (≈ 3.01 M processed tokens over two epochs, DERIVED from the ledger):

| Rate | Minutes | vs v0.2 (202.7 min) |
|---|---|---|
| 248 tok/s (v0.2 median, MEASURED) | 202.6 | 1.0× |
| 600 (minimum viable) | 83.7 | 2.4× |
| 1,200 (target) | 41.9 | 4.8× |
| 2,000 (stretch) | 25.1 | 8.1× |

Mode L, Mode Q and ablation runs are short continuations or slice-sized tests (7.10), not full re-trainings.

### 13.6 What would move the numbers

| Lever | Direction | Known size | Decided by |
|---|---|---|---|
| ε of the int8 and bf16 GEMMs | the largest single uncertainty | table above | P-K1, P-K2 |
| int8 forward adopted | up to 1.5× on the eligible 52 % | table above | P-K2, P-Q6 |
| Overclock | + 12 % in tensor rates | table above | P-H3, P-H4 |
| Larger step (16 k–32 k tokens) | removes any stream/host exposure; costs boundary RAM | 5.3, 5.5 | P-T12 |
| Copy-engine overlap absent | stream and boundary traffic add to compute instead of hiding | up to + 1.5 s per step | P-H12 |
| DeltaNet backward slower than planned | the chunk-state sweep and backward are serial chains | 1–2 % planned | P-K8, P-K10 |
| WSL2 vs Windows-native | the engine saw +24–28 % decode on Windows-native; training effect unknown | — | P-H11 |
| Pad compaction | up to 2.6 % tokens saved | 5.7 | P-T13 |
| Mode Q | + a few percent per step | 4.10 | P-E8 |
| Mode L | + 0.2 % FLOPs | 7.12 | — |

---

## 14. Risks and kill criteria

A kill criterion is a number or event, fixed *before* the work, that ends a line of effort and records the result. The aim is that no part of this project can keep consuming time because it is still "promising".

### 14.1 Risk register

Likelihood and impact are qualitative ESTIMATES (L = low, M = medium, H = high); the early signal and the probe are what make them testable.

| # | Risk | L / I | Early signal | Probe | Mitigation | Kill criterion |
|---|---|---|---|---|---|---|
| R1 | GEMM efficiency ε far below 0.4 | M / H | Hand kernels < 0.85 × cuBLASLt; ε < 0.35 | P-K1, K2 | Redesign tiles before more kernels; dequant-to-scratch + bf16 GEMM fallback | If both paths give < 450 tok/s at T = 8 192 in the bench, re-base the targets and tell the author before building T7+ |
| R2 | DeltaNet backward wrong or slow | M / H | Gradient error > 1e-3 vs fp64; chunk sweep > 2 % | P-K8–K10, P-T1 | fp64 sequential oracle, unfused chunk path as fallback | Backward not within tolerance after the oracle-guided fix → use the unfused path and accept the speed cost |
| R3 | Silent bit flips on the overclocked, non-ECC card | M / H | Canary hash differs; first-step rerun differs | P-H4, P-H10 | Canaries (15.3), run clocks stepped down | Any canary failure in a run → that run is void; two in a week → stock clocks for training |
| R4 | The served model is not one numeric function | H / M | P-Q4 spread large; CPU vs GPU KL large | P-Q4, S6, S7 | Gates set from the measured spread; Mode B if the CPU residual matters | None (this is a property to measure, not to kill) |
| R5 | The core's extra bytes do not fit beside the 7.5 GiB bank (headroom 0.16 GiB at the default prefill micro-batch, none at `-ub 2048`, at the 10.7 GiB ceiling on the engine's whole-card basis; 16.3.1) | H / M | P-E4 peak > 10.7 GiB with the real merged core | P-E4, P-X5 | Byte-budget solver (D22), Mode Q (D23) | If no assignment passes P-Q3 inside Δ_max and Mode Q fails P-E8, the adapter-only export (E-A) is the release and the engine cost is reported |
| R6 | Streaming/host contention exposes the PCIe time | M / M | P-H6 slowdown (measured worst 0.87–0.90 of the unloaded rate); P-H12 overlap lost (measured: it exists only with the flush rule of 5.4) | P-H5, H6, H12, T7 | Flush rule (D27); bigger step, bf16 boundary store, serialized streams (solver) | If stall > 10 % at 32 k tokens after the solver, reduce the target to the measured rate |
| R7 | Pinned memory > 20.4 GiB unavailable on WSL2 | M / M | **Closed 2026-10-24:** P-H7 held 32 GiB on WSL2 and on Windows-native (limit not found); P-M3 held 24 + 4 GiB for 30 minutes | P-H7, M3 | bf16 boundary store, smaller step, pageable boundary store | None; the plan degrades gracefully (5.3) |
| R8 | Tokenizer/template drift from the served model | L / H | P-D1/D2 mismatches | P-D1, D2 | libllama for both; any mismatch blocks training | Any mismatch → stop (no workaround) |
| R9 | MTP wiring wrong, MTP co-training harms acceptance | M / M | P-S5 fails; P-T8 acceptance falls | P-S5, T8 | Keep the block frozen unless P-S5 passes | Acceptance below the untrained value → MTP off |
| R10 | Merge erases the delta (Case 1) at the chosen types | M / H | P-Q3 retention < 0.98; P-E7 ≈ 100 % idempotent | P-Q3, E7 | Higher type inside budget; Mode Q | See R5 |
| R11 | Mode L costs quality or does not move hit rate | H / L | G-L1/G-L2 fail at two router learning rates | P-L1–L10 | Gated, off by default; C0 is always exported | Close Mode L with the negative result written down |
| R12 | Zero-spill on the full-attention layers is infeasible or counter-productive | H / L | P-E1: 100 % residency drops overall hit to 0.711; P-E3: margins too small | P-E1, E2, E3 | Layer weights stay uniform unless P-E2 shows a cost difference | If P-E3 shows the needed logit shift is beyond the trust region, close Mode L-Z |
| R13 | Engine layout needs differ from the standard GGUF layout | M / M | P-E10 passes only with padded/aligned layouts | P-E9, E10, E11 | Layout manifest + optional engine-pack plug-in (16.5 B4) | None; the manifest makes the decision data-driven |
| R14 | CPU/GPU numerics diverge on outlier tokens | L / M | P-E12 correlation present | P-E12, E13 | Mode B / activation-outlier regulariser, gated | Adopt only if its own gate passes; otherwise closed |
| R15 | A dependency changes under us (driver, CUDA, llama.cpp) | M / M | Parity probes drift | P-Q4, P-S1 | Pins: driver 617.42, nvcc 13.0.88 (avoid toolkit 13.2), llama.cpp `2ca15f540` | Re-run Wave 0 parity set after any change |
| R16 | Train-serve gain is smaller than hoped | M / M | P-X2 shows the old adapter already served well | P-Q3, P-X2 | We claim nothing before measuring | None; the claim is only made from measurement |

### 14.2 The 12 parked problem areas — disposition

From the project runbook (2026-10-21 review of the proposal files). None was started at the time; each has a home in this document.

| # | Parked item | Disposition here |
|---|---|---|
| 1 | Plan used 432 MiB per expert layer; the file averages ≈ 465 MiB (18.16 GiB / 40) | **Resolved by data.** Layers are 432 MiB (Q4_K down) or 498 MiB (Q6_K down) (READ-SRC, 2.3); arenas are sized per layer from the GGUF (5.3, T1) |
| 2 | Unit and arithmetic slips (421.8 vs 432 MiB; 17.4 vs 16.5/16.1 ms; 15.19 vs 16.03 GiB per probe token) | **Resolved.** Units fixed in 0.2; all arithmetic regenerated by script (Appendix E); 432 MiB / 26.1 GB/s = 16.5 ms is replaced by the per-layer figures of 5.4 |
| 3 | The 0.74 tok/s probe speed has no source | **Dropped** (2.9); the baseline is the measured 247.85 tok/s median |
| 4 | "3.2–4×" claim | **Replaced** by the sensitivity table (13.3) and targets (1.4); the claim is UNVERIFIED |
| 5 | AVX-512/AMX/DDR5 assumed; the machine is AVX2/DDR4 | **Corrected** (2.1); the CPU-compute training route (KTransformers style) is closed on this machine |
| 6 | "4-bit trained beats Q5" and "0.00 % drift" untested | **UNVERIFIED** (14.3). "0.00 %" holds only for spliced bytes (P-X1); the KL step is P-X2 |
| 7 | Q4_K nibble-layout text wrong (32-byte chunks, not 8-byte pairs) | **Corrected** (2.9, 6.2); verified by P-Q1 against ggml |
| 8 | Cold-tail and sparse-DMA assumptions need a *training-time* routing trace; the engine trace is decode-only | The routing tape (D7) is recorded in `eval` and in training; Mode L uses both (7.12); sparse DMA is not planned |
| 9 | Scrub persona names, handles and paths from the proposal files before use | **Done by rule:** nothing from those lines is carried over (0.5); a privacy scan runs before any publication (15.6) |
| 10 | 13 reviewer bug claims unverified here | Kept as a trap checklist for the native code (Appendix C); versions of the Python oracle pinned |
| 11 | Dense 32–70 B and 16–24 k context claims, no non-MoE test | **Out of scope** (Chapter 3); longer windows are a memory re-plan (P-M2 shows headroom) |
| 12 | Write modules 1–3 after the engine, reusing `cold_weights_view` | Stage T1 (Chapter 12); `cold_weights_view` is reviewed for reuse (provenance header "adapted from pelias-engine, licence Apache-2.0"); code starts only on the author's go |

### 14.3 Register of UNVERIFIED and ESTIMATE items

Every item here is **not relied on** until its probe passes.

| Claim or number | Where | Probe | If it fails |
|---|---|---|---|
| ε in 0.4–0.6; ε₈ = ε − 0.1 | 1.4, 13 | P-K1, K2 | Re-base targets |
| int8 tensor cores 4× bf16 on this card | 1.2, 4.5 | P-H3 | **Passed 2026-10-24:** int8 259.5 vs bf16 65.2 = 3.98×, sustained, both platforms |
| FP8 behaviour | 4 | P-Q10 | Stays unplanned |
| Proposals' 3.2–4× speed-up | 2.9 | none (dropped) | — |
| Reviewer claim A2 ("catastrophic bf16 summation") | 2.6 | P-Q8 | Already judged overstated |
| Served attention uses f16 internally | 2.5 | P-Q7 | Choose our internals by measurement |
| FA backward at head-dim 256 fits `sm_89` tiles | 6.6 | P-K7 | Blocked fallback |
| Integer MMA type combinations compile as planned | 6.2 | P-K2 | bf16 path |
| IMRoPE pairing convention | 6.6 | P-S4 | Fixed by the probe |
| MTP wiring/alignment | 7.6 | P-S5 | MTP stays frozen |
| Size of v0.2's ΔW (3 % σ assumption) and the rounding-noise model | 4.3 | P-Q3 | Types chosen from data |
| Served base sits exactly on its grid (erasure model) | 4.3 | P-E7 | Expectations in P-Q3 change |
| Pinned > 20.4 GiB on WSL2 | 5.3 | P-H7 | **Passed 2026-10-24:** 32 GiB held on WSL2 and Windows-native (the limit above 32 GiB is not found) |
| Copy engines overlap H2D/D2H | 5.4 | P-H12 | **Measured 2026-10-24:** overlap exists only with a flush after each enqueue (0.92 all three, 0.84 pair); the streamer flushes (D27) |
| CUDA graphs work on this WSL2 | 6.9 | P-H9 | **Passed 2026-10-24:** exact over 10⁵ replays, 68–85 % launch saving, both platforms |
| v0.2's GGUF experts equal a fresh `Q4_K_M` | D9 | P-X0 | Base chosen by hash result |
| Definition of v0.2's tok/s | 0.2 | P-T10 | All comparisons re-based |
| Router-locality training works at 35 B / 256 experts | 7.12 | P-L4, L5 | Mode L closed |
| *Sticky Routing* (2607.08780) findings | 2.8 | none (unreadable) | Not relied on |
| Full-attention layers need priority residency | 16.3 | P-E1 (hit rate: refuted), P-E2 (cost: open) | Layer weights stay uniform |
| Byte-load kernels explain the 9.59 vs 26.2 GB/s gap | 16.3 | P-E10 | Engine finds another cause |
| 2 MiB-aligned host memory is required | 16.3 | P-E11 | Alignment optional (as measured on WSL2 for CPU reads) |
| Δ_max = 0.16 GiB (default micro-batch) / none (`-ub 2048`) at the 10.7 GiB ceiling, whole-card basis (≈ 1.2 / 0.9 GiB on the process-only reading) holds for the engine's own allocator | 16.3.1 | P-E4 | Solver re-run with the measured value |
| A miss costs more on full-attention layers than on DeltaNet layers | 16.5 B1 | P-E2 | Layer weights stay uniform |
| Windows-native decodes 25–28 % faster than WSL2 (two Windows launches vs one WSL launch) | 16.3 X9 | P-H11 (trainer), the engine's three-launch repeat | Platform stays WSL2 |
| Q6_K 210-byte blocks force padding or a repack for 16-byte reads | 16.5 B4 | P-E10 | Engine-owned layout; the standard GGUF is unchanged |
| FP16/BF16 accumulators cause CPU/GPU divergence | 16.5 B5 | P-E12, E13 | Mechanism is activation quantization instead |

### 14.4 Decision triggers at a glance

| Decision | Trigger | Action |
|---|---|---|
| Mode R vs F | P-K2 ≥ 1.3× and P-Q6 KL(F) ≤ KL(R) | F default, else R |
| Core types | P-Q3 inside Δ_max (P-E4) | Solver output; else Mode Q; else E-A |
| Router locality | P-L1 gap > 0.03 and P-L2 shows a cost curve worth beating | Build Mode L; else closed before any code |
| Zero-spill layers | P-E2 shows a cost difference **and** P-E3 shows feasible margins | Layer weights; else uniform |
| Mode B / N | P-E12 criteria met | Build; else closed |
| Platform default | P-H11 | Faster platform |
| Release artefact | Rung 7–8 gates | Merged GGUF if both pass; adapter-only otherwise |

---

## 15. Safety rails

These are rules the code enforces, not intentions.

### 15.1 VRAM governor

* Hard ceiling **10.7 GiB** (author ruling 2026-10-23, lowered from 10.9: P-H8 measured 10.83–10.85 GiB free once the CUDA context exists, so 10.9 was not reachable). Every device allocation goes through an arena that is registered with the governor at plan time; the plan's sum is checked against the ceiling before the first `cudaMalloc`, and a plan above it is **refused**, never trimmed silently. **What the number is measured against is open (16.7, decision 7):** the plan sum and `cudaMemGetInfo` see the process's own allocations, while the runtime NVML check below sees the whole card, which includes Windows' share (0.89–1.01 GiB before the process starts, MEASURED). Decision 7 was ruled on 2026-10-24 (the ceiling comes from the probe at start, 3.3 U2); until the start-up rule and the margin test exist, both checks run, so the effective limit is the stricter of the two.
* At start the run refuses unless the smaller of CUDA free and NVML free, after the context exists, is at least the ceiling (P-H8; margin 0 by the author's decision: the measured free is 10.83–10.85 GiB, so the ceiling leaves 0.13–0.15 GiB to the physical limit, and a Windows app that grows by more than that stops the start).
* **Ruled 2026-10-24 (decision 7, D28):** the number the start rule compares against is measured, not typed: smaller of CUDA free and NVML free after the context exists, minus a margin. The margin is set by the disturbance test (11.12); 10.7 GiB is this machine's value until then. A runtime log of NVML used and CUDA free (peak and floor per block of steps) is the way to that margin and is the training-side half of the "flush" question (3.3, U5).
* At runtime the governor re-reads NVML used-memory every N steps; exceeding the ceiling aborts the step cleanly (checkpoint first), it does not continue.
* Zero `cudaMalloc` in the step loop (P-M5): fragmentation and surprise growth cannot occur.

### 15.2 Host memory

* The planned footprint (28 GiB at 16 k tokens/step, 33 GiB at 32 k, 5.3) must fit under WSL2's 48 GiB cap with headroom; P-H7 and P-M3 prove it before any real run.
* Pinned allocations use `cudaHostAlloc` (the form measured to work to 32 GiB, P-H7); `cudaHostRegister` is avoided where pointer identity matters. The engine saw a `memlock` limit of 65,536 KiB that did not stop `cudaHostAlloc`; the allocator logs the limit it sees.
* The fallback chain when pinned memory runs short: bf16 boundary store → smaller step → pageable boundary store → fewer tokens per window. Each fallback is logged.

### 15.3 Canaries — the card has no ECC and runs overclocked

* **Kernel canary:** every 50 steps a fixed small GEMM + reduction runs with known input and a stored hash; a mismatch marks the run suspect, writes a checkpoint, and stops.
* **First-step rerun:** the first optimizer step of every run is executed twice and compared bit for bit (deterministic mode); differences stop the run.
* **State checksum:** a hash of the LoRA masters and optimizer state is stored with every step in the metrics; resume verifies it.
* **Finite checks:** loss, gradient norm and per-layer activation norms are checked for NaN/Inf every step at essentially no cost; on failure the step is replayed from the stored boundary, the first bad layer is reported, and a second failure stops the run.
* **Clock record:** NVML clocks, power and temperature are logged per step; the throttle-reason field is logged too but not trusted (it reads "gpu_idle" under full load on both platforms, P-H4); sustained throttling is judged from the SM clock falling below the run clock and from temperature, and is a stop condition.

### 15.4 Atomic checkpoints

* A checkpoint is written to a temporary directory, every file is flushed (`fsync`), a manifest with SHA-256 per file is written last, and only then is the directory renamed into place. A crash leaves either the previous checkpoint or the new one, never a half-written one.
* Four registries (weights, optimizer, data cursor/RNG, MTP; 7.8) are versioned together by one manifest. The last three checkpoints are kept; nothing is deleted by the trainer (15.5).
* Resume verifies the manifest, then proves bitwise continuity (P-T11).

### 15.5 Files, deletion and overwriting

* The served GGUF and the base files are opened read-only and are never overwritten. Every export writes to a new timestamped directory.
* The exporter needs ≈ 21 GB free on the destination and checks before starting; it writes to a temporary name and renames only after the hash proof (9.3).
* The trainer never deletes. Cleanup is a separate, explicit author action. On the device bridge, deletion needs the author's permission and, if refused, files move to a `_to_delete/` folder instead.

### 15.6 Privacy rails

* Dataset text, persona/identity text, the GGUF chat template and private equations are never printed, logged or committed. Reports carry counts, lengths, ids and pass flags only.
* The Hugging Face token is never read, printed or passed to any tool.
* Before any document or log leaves the machine (including to the repositories), a scan checks for the category of strings the redaction rules name (account names, model filenames, local paths, persona/identity words, tokens/keys, dataset fragments). A hit blocks the publication and is reported by category and count, not by content.
* The public documents make no "first of its kind" claim and do not copy AGPL/LGPL code.

### 15.7 WSL2 specifics

* Everything is launched from WSL, except the Windows-native probe runs, which the author launches from the one PowerShell script permitted on 2026-10-23 (read-only facts, then build and run). No other use of PowerShell.
* `/dev/dxg` is the only GPU device in the guest; ReBAR cannot be inspected from inside it with the usual tools, but NVML (used by P-H8) reports BAR1 total 16,384 MiB on both platforms (Resizable BAR on; used 0 MiB inside WSL2, 16,355 MiB on Windows-native; meaning not examined; no on/off comparison exists). THP made no measurable difference in the engine's tests (< 1 %), so it is left as found.
* Files on the NTFS mount read at ≈ 0.5 GB/s (engine measurement), so the GGUF is copied once to the WSL ext4 disk or read once into the pinned store; P-M4 measures both.
* Windows updates, sleep and hibernation will kill a long run; the run keeps checkpoints cheap and the manifest consistent, and the runbook lists the settings to check before a run.
* `.wslconfig` caps (48 GiB RAM, 48 GiB swap on D:) are recorded in the machine-state file at the start of every run.

### 15.8 Stop conditions (the watchdog)

| Condition | Action |
|---|---|
| Plan above 10.7 GiB; free VRAM below start threshold | Refuse to start |
| Runtime VRAM above ceiling | Checkpoint, abort step, stop |
| Canary mismatch; rerun mismatch | Checkpoint, mark run suspect, stop |
| NaN/Inf twice at the same step | Stop with the first-bad-layer report |
| Sustained throttle (> 60 s) | Checkpoint, stop |
| Free disk below the export gate | Do not export |
| Any gate failure in `verify` | Stop the pipeline; waiver needs a dated line in `decisions.md` |

### 15.9 Reproducibility pins

Driver 617.42 (CUDA 13.4), nvcc 13.0.88 (toolkit 13.2 avoided), `sm_89` + PTX, llama.cpp commit `2ca15f540` (b11399) with the carried patches `mm_ids_dedup`, `quantize_signed_id`, `cuda_hole_fix`, `pin_cold_to_cpu`; all pins recorded in every result file's machine-state hash.

---

## 16. Engine co-design audit — locking this document to pelias-engine

This chapter exists because the engine's measurements arrived after the first draft of the trainer plan. It does four things: it audits this document requirement by requirement against the author's five engine invariants (16.2); it lists every assumption that drifted and what was done about it (16.3); it reconciles an outside research report with the engine's own measurements (16.4); and it gives the concrete bridges — loss formulations, export steps, gates — for what was missing (16.5). The data still to be derived is collected in one place, with method, use and location in this document (16.6). The edits proposed for the engine blueprint, the two-way contract and the open decisions are in 16.7.

### 16.1 Sources, and what "locked" means

| Source | Date | Status in this chapter |
|---|---|---|
| Engine blueprint (`PELIAS_ENGINE_LINUX_BLUEPRINT.md`) | 2026-10-22 | Plan; file names in it are proposals, nothing is built |
| Engine results ledger (`benchmarks/RESULTS.md`, sections 1–9.4) and `raw/` | to 2026-10-23 | MEASURED / DERIVED as labelled there |
| Engine machine state (`raw/e0-2026-10-22/machine_state/`) | 2026-10-22 | `wsl --version` WSL 3.0.4.0, kernel 6.18.54.1-1; `.wslconfig` has `transparent_hugepage=always swiotlb=65536`; GPU overclocked (memory +1000 MHz, core ≈ 2790 MHz); HAGS off |
| Trace re-analysis run for this document (P-E1) | 2026-10-23 | DERIVED, read-only on the engine's recorded routing; baseline reproduced exactly |
| The author's five invariants | 2026-10-23 14:59 | Requirements (16.2) |
| An outside, AI-generated research plan ("tri-environment hardware-verified research plan") | 2026-10-23 | **Input to be checked, not a source.** Claim-by-claim in 16.4 |

"Locked", in the author's message, is treated as follows. **Rulings recorded in the engine repo** (H6 float-spread governs; 10.9 GiB VRAM ceiling, which the author lowered to 10.7 GiB for the trainer on 2026-10-23 while the engine repo still says 10.9; which loose patches are carried) are DECISIONS and are used as given. **The 7.5 GiB bank** is not a ruling in the repo; it follows from the 10.9 GiB ruling and the measured peaks (10.40–10.69 GiB at 7.5 GiB, 11.0–11.19 GiB at 8.0 GiB) and is the largest 0.5-GiB step under the ceiling (DERIVED; under 10.7 it is still the largest step on the whole-card basis, with 0.01–0.03 GiB to spare at `-ub 2048`). **Everything else in the invariants is a requirement to be met or a premise to be checked**, and 16.2 says which.

### 16.2 Compliance matrix — requirement by requirement

Status key: **C** compliant as already specified in this document (kept unchanged) · **F** the document drifted or was wrong and has been corrected · **A** specification added by this audit · **U** the premise is not supported by the engine's data; handled as an optional mode or recorded as a non-requirement · **N** not built, by design · **O** open on a named probe.

| # | Requirement (from the invariants) | Where | Status | Note |
|---|---|---|---|---|
| R1 | The trained core's extra bytes fit beside the 7.5 GiB bank under the 10.7 GiB ceiling | 9.4, 16.5 B2 | **F** | The first draft defaulted to an all-Q8_0 core (+0.59 GiB). Headroom is 0.16 GiB at the engine's default prefill micro-batch and none at `-ub 2048` (DERIVED; whole-card basis, 16.3.1; it was 0.36 / ≈ 0.11 GiB at 10.9). Replaced by a byte-budget solver (D22) and Mode Q (D23) |
| R2 | Bank residency is encouraged **in training**, not by inference-time clamping | 7.12 | **C** | Mode L was already specified when the audit arrived |
| R3 | The capacity loss uses the engine's actual resident set `R_l` (the `PELIAS_HOT_SET` content) | 7.12 item 2 | **C** | Extended by the zero-miss hinge, the clamped-forward variant and distillation (16.5 B1) |
| R4 | 100 % VRAM hit on the 10 full-attention layers | 16.5 B1 | **U** | The engine's traces show these layers route like the rest (hit 0.8299 vs 0.8265); making them fully resident costs 4.605 of the 7.5 GiB and lowers the overall hit rate from 0.8273 to 0.7114 (P-E1). Offered as an optional, gated mode (L-Z) with its own kill criterion; layer weights stay uniform unless P-E2 shows a cost difference |
| R5 | Expert consolidation as an alternative to a capacity loss | 16.5 B1.6 | **N** | The experts are frozen 4-bit bytes (D2, D7). Consolidation needs trainable experts and an exact expert quantizer. Not built; the reopen condition is written down |
| R6 | The trainer emits a `PELIAS_HOT_SET` profile identical to the engine's `export_hot_set.py` | 16.5 B4 | **A** | C++ writer (D25), parity probe P-E14, zero-slot accounting |
| R7 | Train on the served bytes; dequantization bit-exact | D1, D2, 6.2, P-Q1 | **C** | |
| R8 | Re-quantization of any rewritten tensor uses ggml's exact super-block arithmetic (`fp16 d`, `dmin`, packed 6-bit scales, rounding), commit `2ca15f540` | 9.3 step 2, P-E6 | **A** | CPU port of the reference quantizers (MIT), then GPU port; 100 % bit-identical required |
| R9 | Quantization-aware training simulates that exact arithmetic | 4.10 (Mode Q, D23) | **A** | Straight-through estimator through the exact quantizer; exported bytes equal trained bytes |
| R10 | Output tensor types are only those the engine can serve | 9.4, P-E5 | **F** | `Q4_K`, `Q6_K`, `F32`; `Q8_0` only if the engine adds it |
| R11 | The accuracy gate is the stock noise floor (KL ≤ 0.017353 at batch 16 / ≤ 0.018286 at batch 1024) | 9.6, 16.5 B3 | **F** | Those are *stock-vs-stock* spreads on the base model. For a fine-tuned file there are three different KLs with three gates (B3); none compares the fine-tune to the base |
| R12 | Pinned to llama.cpp `2ca15f540` (b11399) | 2.1, 15.9 | **C** | |
| R13 | Tensor starts aligned | 9.3 | **C** | GGUF alignment 32; the served file's 123 expert tensors are all 16-B aligned (P-E9, READ-SRC) |
| R14 | Rows and splits align to `QK_K = 256` | 6.2, 9.3 | **C** | Every expert and core dimension is a multiple of 256 (2048, 512, 4096, 8192); the exporter asserts it |
| R15 | 16-byte (`uint4`) coalesced reads of cold experts | 16.5 B4 | **O** | Q4_K rows are 16-B multiples; **Q6_K rows (420 B) and blocks (210 B) are not**. Whether padding, a repack or a staging kernel is right is the engine's measurement (P-E10) |
| R16 | Expert tensors on 2 MiB boundaries | 16.5 B4 | **A** | Free in a standard GGUF: expert-last layout plus a header pad (no extra bytes; llama.cpp load check in P-E9). Whether it matters is P-E11 |
| R17 | One owner for the packed layout and `repack_gguf.py` | 16.5 B4 | **F** | The engine owns the packed layout and its repacker (its blueprint lists `repack_gguf.py (later)` under *its* tools). The trainer emits the standard GGUF plus a layout manifest and calls the engine's repacker as a plug-in. The first draft implied the trainer would define the layout |
| R18 | After training, the hot set is re-profiled from the trained model's routing | 9.5, 7.12 | **C** | `expert_usage.json` from the tape |
| R19 | Activation outliers are regularized against CPU-branch/GPU-branch divergence | 16.5 B5 | **A** (gated) | Mode B/N (D24), adopted only if P-E12 passes |
| R20 | Router-logit spikes / near-ties do not flip the selected experts between branches | 16.5 B5 | **O** | Measured first (P-E3 extended); a regularizer only if the flip rate is material |
| R21 | The divergence mechanism is FP16/BF16 accumulation | 16.5 B5 | **U** | In the K-quant paths the products accumulate in int32 and the scales in fp32; the branches differ by activation-quantization granularity and fp32 summation order. P-E13 audits the engine's kernels |
| R22 | A separate section lists the data to derive: method, use, place in this document | 16.6 | **A** | |

**Count (rows, unweighted): 22 requirements — C 7, F 4, A 6 (one gated), U 2 plus one non-requirement N, O 2.** Read honestly: about a third of the rows were already satisfied by the first draft as written, another fifth were *wrong or drifted* and have been corrected in place, and the rest are new specification or are not supported by the data. The author's 70–80 % estimate is not confirmed by this count — but there is no natural denominator (a row can be one line or one chapter), and none of the unchanged material was rewritten: the structure (served-bytes training, byte-splice export, verification ladder, routing tape, hot-set hand-off) survived intact; the corrections were concentrated in the byte budget, the Q8_0 default, the citation of the routing paper and the accuracy-gate wording.

### 16.3 Drifted assumptions and discrepancies — register

Every line is something that was assumed, stated, or inconsistent, with what the evidence shows and what was done. "This doc" means the trainer plan; "engine" means the engine blueprint (changes in 16.7).

| # | Assumption or discrepancy | Evidence | Resolution |
|---|---|---|---|
| X1 | An all-Q8_0 core is the safe default (this doc, first draft) | +0.59 GiB vs 0.16 GiB / none free at the 10.7 ceiling (DERIVED from the engine's peaks; 0.36 / ≈ 0.11 at 10.9); engine kernels planned for Q4_K/Q6_K/F32 only | D4 revised; D22, D23; 4.3, 9.4 |
| X2 | The earlier summary of arXiv 2608.18261 ("every setting broke its 1 % gate", in the engine blueprint) | Paper read in full: untrained rerouting costs +0.8…+2.0 % at ≈ 30 % fewer misses and +4…+5.8 % at 50 %; training from scratch at 137 M/340 M never met "≥ 30 % at ≤ +1 %"; but rerouting after locality training costs +0.9 % vs +7.7 % | 7.12 table; engine blueprint scope line to be corrected (16.7). The paper is evidence that **inference-time** clamping is costly and that training changes the price — not that router training is impossible |
| X3 | 2 MiB huge pages are *required* to avoid GPU/CPU TLB trouble | CPU read test: 44.86 GB/s with 2 MiB pages vs 45.81 GB/s with 4 KiB (MEASURED, ledger §1); WSL is already booted with `transparent_hugepage=always`; GPU effect not measured | Optional; P-E11 decides. Not a requirement of the exporter |
| X4 | `uint4` coalescing is the *established* cause of the 9.59 vs 26.2 GB/s gap | The ledger itself calls the byte-load cause "unverified". The checksum kernel (26.2) and the compute kernel (9.59) read the same pinned pages, so page size alone cannot be the cause (DERIVED) | Hypothesis, probe P-E10. The WDDM 4-KiB statement is true (16.4 G2) but does not explain this gap |
| X5 | "KL ≤ 0.017353 / 0.018286 against the oracle" | Stock-vs-stock at batch 16 / 1024 against a batch-8 reference, 16 chunks × 1,024 tokens, on the base model | Three KLs, three gates (16.5 B3); initial values are the engine's numbers; re-measured on the trainer's tokens (P-Q4) |
| X6 | Zero spill on the 10 full-attention layers is needed because they are latency-critical | Hit 0.8299 vs 0.8265 elsewhere; zero-miss 0.3636 vs 0.3602; full residency costs 4.605 GiB and lowers the overall hit to 0.7114 (P-E1). Per-layer latency cost by layer type is **unmeasured** (P-E2) | Optional L-Z; weights uniform until P-E2 |
| X7 | FP16/BF16 accumulator divergence between the AVX2 and CUDA reduction trees | K-quant dot products accumulate in int32, scales in fp32; the real difference is Q8_K (256-element blocks, CPU) vs q8_1 (32-element blocks, CUDA) activation quantization | 16.5 B5; P-E12 measures, P-E13 audits the engine's kernels |
| X8 | Engine blueprint: "CPU read of cold experts 26.6 GB/s" | 26.56 GB/s is the CPU's share while the GPU reads at the same time; CPU alone reads 43.8–45.9 GB/s; the combined 47.62 = 26.56 + 21.06 | Engine table to be corrected (16.7). The ceiling uses 47.6 combined, which already assumes the GPU share at the checksum-kernel rate |
| X9 | Engine blueprint scope: "a Windows-native build is not in scope" | Ledger §9.4: Windows-native decode is 25 % (short) and 28 % (12k) faster than the WSL cell (two Windows launches vs one WSL launch; not yet a claim); the difference is ≈ 5.2 ms per token | Decision for the author (16.7). The trainer re-measures both platforms for the exported model either way (9.5 step 4) |
| X10 | An outside report's Windows figure of "57.0 tok/s (17.54 ms/token)" | Recorded Windows-native runs: 47.3 and 46.8 tok/s short, 53.7 and 54.1 at 12k. 57.0 appears in the logs only as the top of a 3-request range at 4k depth for an 8.5 GiB WSL bank (median 53.7). The author says the number may have been mistyped | Not used. Gap analysis recomputed from the recorded runs (16.4.1) |
| X11 | Engine bank peak "10.40–10.44 GiB at 7.5 GiB" taken as the headroom for all settings | The same bank peaks at 10.67–10.69 GiB at `-ub 2048` | Both stated everywhere (2.7, 9.4, P-E4); the engine blueprint should get a VRAM-budget row (16.7) |
| X12 | Engine blueprint: "After E13 the lessons go back into the pelias-trainer" | The trainer's byte budget, hot-set writer and gates depend on E-phase results now | A two-way contract (16.7): which engine milestone releases which trainer input |
| X13 | An outside report: pinned memory is capped at ≤ 32 GB under WDDM | **Refuted on this machine:** P-H7 held 32 GiB (= 34.4 GB) of `cudaHostAlloc` memory on WSL2 and on Windows-native with nothing swapped in (2026-10-23/24); the engine's earlier 20,448 MiB was the largest size it had tried, not a limit. The limit above 32 GiB is not found | P-H7 raised to `--max-gib` above 32 if more is ever needed |
| X14 | An outside report: lock GPU clocks at 2475 MHz (`nvidia-smi -lgc`) | The card runs overclocked (core ≈ 2790 MHz at 965 mV, memory +1000 MHz); 2475 would be an under-clock, and clock-lock support under WSL is unverified | Not adopted. Clock policy stays with the author; the trainer logs the clocks (15.3) |
| X15 | "The ceiling" is one number with one meaning | Three different quantities carry the name. (a) The engine's ruling (10.9, engine blueprint) is compared with whole-card peaks: NVML used, including Windows' share (MEASURED, engine ledger). (b) The engine's gates say "VRAM at most 11.2 GiB" (P1 and G3, whole card): that is where the 11.2 in the author's memory and in the first P-H8 rule comes from. (c) P-H8, the governor's plan sum and `cudaMemGetInfo` see only the process. The card holds 0.89 / 1.01 GiB for Windows before the process starts and 1.04 / 1.14 GiB once its context exists (NVML used, A / B, P-H8); CUDA free after the context is 10.83 / 10.85 GiB | Decision 7 (16.7), ruled 2026-10-24 for a probe-measured, process-only ceiling. Headroom under both readings is in 16.3.1; the other chapters still print the whole-card figures (the engine's basis, and the stricter) and say so |
| X16 | First P-H8 rule: free after the context ≥ ceiling + 0.3 (11.2 GiB) | Unreachable on this card: CUDA free is 10.83 GiB (WSL2) and 10.85 (Windows-native), the same on every run and in 21 engine logs, so it is a fixed driver budget (inference), not desktop load; NVML used before the context moved by about 0.16 GiB between runs (sign-out, restart) while CUDA free did not move | Rule replaced: the smaller of CUDA free and NVML free after the context ≥ 10.7 (margin 0, a margin can be added); ceiling lowered to 10.7 by the author (D26) |

#### 16.3.1 The 7.5 GiB budget, in one table

| Quantity | Value | Label |
|---|---|---|
| VRAM ceiling | 10.7 GiB | DECISION (author ruling 2026-10-23; 10.9 from 2026-10-22 until then) |
| Bank | 7.5 GiB, 4,170 experts, including one zero slot per layer (72.6 MiB) | READ-SRC (engine hot-set sim) |
| Whole-card peak, default micro-batch | 10.40–10.44 GiB → non-bank 2.90–2.94 GiB → headroom to 10.7: 0.26–0.30 GiB | MEASURED / DERIVED |
| Whole-card peak, `-ub 2048` | 10.67–10.69 GiB → non-bank ≈ 3.2 GiB → headroom to 10.7: 0.01–0.03 GiB | MEASURED / DERIVED |
| Whole-card peak, 8.0 GiB bank | 11.0–11.19 GiB → over the ceiling | MEASURED |
| Δ_max for the core's extra bytes (0.1 GiB margin), whole-card reading (W) | **0.16 GiB** (default micro-batch; 0.20 at the lowest peak) / **none** (`-ub 2048`: −0.09 GiB) | DERIVED; P-E4 re-measures with the engine's own allocator |
| Δ_max, process-only reading (P): the same peaks minus the ≈ 1.0 GiB Windows already holds before the process starts | engine's own use ≈ 9.4 / 9.7 GiB; headroom to 10.7 ≈ 1.3 / 1.0 GiB, ≈ **1.2 / 0.9 GiB** after the margin | DERIVED; the Windows share moved by about 0.16 GiB between probe runs, so P-E4 must record NVML used before the engine starts as well as at the peak |

The "whole card" figure includes the Windows desktop's share, so it is a ceiling-relevant number rather than the engine's own footprint. **Consequences of the two readings (decision 7; ruled 2026-10-24 for the probe-measured, process-only ceiling, with the whole-card figures kept as the stricter planning figures until P-E4 re-measures; 3.3, U2).** On the whole-card reading (W) the earlier conclusions get worse, not better: an all-Q6_K core (+0.28 GiB) fits at neither micro-batch, so classes are raised selectively; for example shared experts + attention + DeltaNet-out to Q6_K is +153.1 MiB (25.7 + 65.5 + 61.9, Appendix E), which fits the 0.16 GiB (163.8 MiB) at the default micro-batch. Taking the rest from the bank costs hit rate: the shortfall of an all-Q6_K core is 0.12 GiB at the default micro-batch and 0.37 GiB at `-ub 2048`, which at the engine's exchange rate (−0.027 hit per −0.59 GiB, ≈ +1 tok/s per +0.01 hit) is about −0.005 hit (−0.5 tok/s) and −0.017 hit (−1.7 tok/s) (DERIVED by interpolation; P-E4 and P-X5 measure it). On the process-only reading (P) the room is about 1 GiB and the 8-bit core (+0.59 GiB) would fit at both micro-batches. The author's choice decides which of these designs is built.

### 16.4 The outside research report — what was checked and what happened to each claim

The report was produced by another AI system from the engine blueprint and the author's description. It is useful as a **list of hypotheses**; as a source it fails its own claim of being "zero-hallucination": its reference numbers in the body are off by one against its reference list throughout (for example the AMD guide is cited as [Ref 5] in the text and numbered 6 in the list; the THP kernel document is [Ref 6] in the text and 7 in the list), and it attributes the engine's own measured 9.59 → 26.2 GB/s figures to the NVIDIA Ada whitepaper. Each claim below was checked against the engine ledger and, where possible, against the cited primary source.

| # | Claim | Check | Verdict | Where it went |
|---|---|---|---|---|
| G1 | Windows-native 57.0 tok/s (17.54 ms/token); gap 6.67 ms (167 µs/layer) | Recorded Windows runs 46.8–47.3 short, 53.7–54.1 at 12k (ledger §9.4); X10 | **Not supported as stated** | Recomputed in 16.4.1 |
| G2 | WDDM `VidMm` manages system memory only in 4-KB pages | Fetched the cited Microsoft page: *"System memory is always managed in 4-KB pages, while memory segments can be managed at either 4 KB or 64-KB granularity as determined by the KMD."* (section "Physical page size"); the page mentions no 2 MB option | **True** | Context for P-E11. It does not show that 4-KB mappings limit the engine: the checksum kernel already reads those pages at 26.2 GB/s |
| G3 | WSL 3.0.4, kernel 6.18.54.1 | Matches the pasted `wsl --version` (machine state). The release notes could not be confirmed (the page fetched showed older releases only), so the PR numbers (#41904 "perf removed", #41888 stack size) are unchecked | **Version true; details unverified** | The engine already plans in-process CUDA-event timing; no change |
| G4 | The 2× prefill difference between load modes comes from 4-KB guest pages; `madvise(MADV_HUGEPAGE)` fixes it | `--load-mode none` is the **fast** mode and is ordinary anonymous memory (4-KB pages, per the engine's audit-script comment); WSL has `transparent_hugepage=always`; the engine's own planned test is to read `AnonHugePages` while each mode serves | **Unsupported; contradicted by the direction of the evidence** | Engine unknown "cause of the load-mode effect" kept; P-E11 |
| G5 | Windows: allocate the CPU cold bank with `MEM_LARGE_PAGES` (needs `SeLockMemoryPrivilege`), the small staging buffer with `cudaHostAlloc` | API facts are as documented; benefit unmeasured; the CPU gain from 2 MiB pages on WSL was nil | **Plausible, unmeasured** | Engine test E0.6-style, only if Windows-native is adopted (X9) |
| G6 | Pin 8 workers to the 8 physical cores (1T) rather than 16 SMT threads; Zen 3 partitions load/store/retire queues in 2T mode (AMD guide §2.14, Table 3) | §2.14/Table 3 exist; the extracted text cannot show the classification of those three queues; the retire queue "can hold up to 256 macro ops or 128 per thread in SMT mode" (§2.10.3). On stock llama.cpp 5–7 generation threads beat 10 (ledger §8.8) | **Partly verified; test it** | Engine E4 A/B (8 pinned vs 16) |
| G7 | Use acquire/release, not seq_cst, for handoff flags; after a `LOCK` avoid FP instructions (AMD §2.15) | §2.15 quote verified exactly: *"Following a LOCKed instruction, refrain from using floating point instructions as long as possible."* On x86 a seq_cst store compiles to `XCHG` (locked); an acquire load and a release store are plain `MOV` | **True; benefit unmeasured** | Engine coding rule where the memory model allows it; A/B in E6 |
| G8 | `PREFETCHNTA` for cold streaming reads to spare the 96 MB victim L3 | Mechanism plausible (L3 is a victim cache); unmeasured | **Hypothesis** | Engine E4 A/B |
| G9 | `uint4` loads staged through shared memory lift zero-copy compute reads from 9.59 to 26.2 GB/s | The engine's own E0.5/E5 item; mechanism plausible; not measured; the citation in the report (Ada whitepaper) is wrong | **Hypothesis (already on the engine's list)** | P-E10 |
| G10 | Pin 16–24 MB of the 36 MB L2 for spine activations and DeltaNet state (`cudaAccessPolicyWindow`) | The API exists; the achievable persisting size on this card under WDDM/WSL is unmeasured | **Hypothesis** | Engine E5 A/B (query `cudaDevAttrMaxPersistingL2CacheSize` first) |
| G11 | Hold `/dev/cpu_dma_latency = 0`; lock GPU clocks | The first applies to native Linux; X14 for the second | **Native Linux only / not adopted** | Engine E1 only |
| G12 | Native Linux projected at 72–86+ tok/s | No measurement exists; the engine's E1 phase is the measurement | **Not used** | — |
| G13 | Gate thresholds (≥ 22 / ≥ 25 GB/s zero-copy; ≥ 27 GB/s cold; handoff p99 ≤ 12 µs native) and the calendar | Invented by the report; the engine blueprint has its own gates and the author sets the dates | **Not adopted** | Candidate gates may be set after the E0/E1 floors |
| G14 | RTX 4070: 46 SMs, 5,888 cores, 192-bit bus, 504 GB/s, 36 MB L2, compute capability 8.9 | Consistent with published AD104 data and with this document's 2.1 | **Consistent** | — |
| G15 | Pinned host memory capped at ≤ 32 GB | X13 | **Unverified** | E11 / P-H7 |
| G16 | WSL2 runs at 44–54 tok/s short and 43–49 at 12k | Engine ledger §8.2: 53.5 / 53.0 / 49.0 tok/s with the 7.5 GiB set, bypass 32, `-ub 2048`. The reference cell A (7.0 GiB, bypass 32, `-ub 4096`, timer off) gives 37.9 / 42.1 | **Supported for the best recorded cell, not for the reference cell** | 16.4.1 (row added). The 5.1–5.2 ms Windows–WSL difference is a like-for-like figure on the reference cell only |
| G17 | The `/dev/dxg` VMBus costs 25–35 µs per layer (1.0–1.4 ms per token) and explains why Windows is faster | Arithmetic: 1.0–1.4 ms is 19–27 % of the 5.1–5.2 ms measured difference on the reference cell. The 57.0 baseline it explains is itself unsupported (G1) | **Unmeasured; at most a fifth to a quarter of the measured gap** | Engine handoff probe (the ledger has 94 µs sync against 13.3 µs resident-doorbell handoff on WSL) |
| G18 | Windows-native: pin workers to physical cores, turn off power throttling (`SetThreadInformation`), `timeBeginPeriod(1)`, to cut launch-to-launch spread (stated as 5–21 %) | The calls exist as documented; the spread figure was not checked against the ledger here; no Windows-native thread-pinning run exists | **Hypothesis, Windows-native only** | Engine, only if Windows-native is adopted (decision 4 in 16.7) |

#### 16.4.1 The gap to the ceiling, recomputed from the recorded runs

The ceiling model gives 10.87 ms per token (92.0 tok/s) with the checksum-rate GPU read and 11.34 ms (88.2 tok/s) with today's 9.59 GB/s compute kernel (engine ledger §4, blueprint). Per-layer figures divide by 40.

| Run (recorded) | tok/s | ms/token | Gap to 10.87–11.34 ms | µs per layer |
|---|---|---|---|---|
| WSL cell A, short | 37.9 | 26.39 | 15.0–15.5 ms | 376–388 |
| WSL cell A, 12k | 42.1 | 23.75 | 12.4–12.9 ms | 310–322 |
| Windows-native r1 / r2, short | 47.3 / 46.8 | 21.14 / 21.37 | 9.8–10.5 ms | 245–262 |
| Windows-native r1 / r2, 12k | 53.7 / 54.1 | 18.62 / 18.48 | 7.2–7.8 ms | 179–194 |
| WSL best recorded cell (ledger §8.2: 7.5 GiB set, bypass 32, `-ub 2048`), short / 4k / 12k | 53.5 / 53.0 / 49.0 | 18.69 / 18.87 / 20.41 | 7.4–9.5 ms (7.35–7.82 for the short run, 9.07–9.54 for the 12k run) | 184–239 |
| *(the report's figure, for comparison)* | *57.0* | *17.54* | *6.2–6.7 ms* | *155–167* |

The §8.2 row uses the same ceiling as the others; that ceiling was not recomputed for a 7.5 GiB set. The §8.2 speeds come from the gauntlet run (ledger §8.2), the other rows from the author's streaming client (ledger §9.2); two measuring clients were used, so comparisons across that line are indicative, not exact (DERIVED, caveats).

The Windows–WSL difference is 5.1–5.2 ms per token at both depths (DERIVED), about the engine's measured timer-free sync-and-copy handoff on WSL (≥ 4.6–5.0 ms). That difference is measured on the reference cell only. On WSL, the §8.2 cell's short-context run (53.5) is 15.6 tok/s above the reference cell's (37.9), more than the whole Windows–WSL difference on the reference cell (47.3 against 37.9), so the cell's settings appear to matter more than the operating system. At 12k the order is different: Windows reference cell 53.7 / 54.1 against WSL §8.2 49.0. Windows-native has not been run in the §8.2 configuration, and the two sets used different measuring clients, so the OS effect at equal settings is unmeasured (DERIVED comparison of recorded runs). The ledger notes that three launches per side are still needed, so this is a consistent pattern, not a result. If the author has a newer Windows run than those in the repository, it should be added to the ledger with its raw file before it is used anywhere.

### 16.5 The bridges

Five bridges close what 16.2 marked **A**, **F** or **O**. B1 is the training-side answer to "make the router use the resident experts"; B2 is the byte budget; B3 is the accuracy gate; B4 is export alignment and the hot-set profile; B5 is the dual-branch numerics.

#### B1 — Bank-aware training (amends 7.12; gates are new)

**B1.1 What "100 % hit" can and cannot mean.** Notation: layer l, token t, router logits z_{l,t} ∈ ℝ²⁵⁶ (F32), router probabilities p = softmax(z), resident set R_l (size k_l, per-expert bytes s_l ∈ {1.6875, 1.945} MiB), top-8 selection S_{l,t}. Per-pair hit h_l = P(e ∈ R_l | e ∈ S). Zero-miss share Z_l = P(S ⊂ R_l). "100 % hit on layer l" means Z_l = 1. There are only three ways to get it:

| Route | What it does | Allowed? | Cost on this model |
|---|---|---|---|
| Capacity | Make R_l all 256 experts | Yes (no training) | The 10 full-attention layers alone take **4.605 GiB** of the 7.5 GiB bank; overall hit falls from **0.8273 to 0.7114** (P-E1, DERIVED) |
| Concentration | Train the router (and the core around it) so the top-8 falls inside a smaller R_l | Yes — this is what the trainer can do | Unknown: the quality cost is the question P-L4/P-E3 answer |
| Clamp | Bias or mask the router at inference | **No** (arXiv 2608.18261: +0.8…2.0 % perplexity at ≈ 30 % fewer misses, +4…5.8 % at 50 %, on an untrained model) | — |

The cost of concentration is in how many experts a layer needs to cover its own routing. From P-E1, on the engine's recorded routing (84,503 fit and 36,657 held-out decode steps), sets ordered by fit frequency:

| Set size k per layer | GiB if all 10 full-attention layers held at k | Untrained all-8-inside share, full-attention layers | Other layers |
|---|---|---|---|
| 96 | 1.73 | 0.305 | 0.324 |
| 128 | 2.30 | 0.519 | 0.517 |
| 192 | 3.45 | 0.831 | 0.817 |
| 224 | 4.03 | 0.941 | 0.931 |
| 256 | 4.61 | 1.000 | 1.000 |
| engine baseline (water-filled, 1,045 experts ≈ 104.5 per layer) | 1.88 | 0.364 | 0.360 |

An untrained router needs 210 / 227 / 247 experts per layer for the all-8-inside share to reach 90 / 95 / 99 % (full-attention layers; the others are within one expert). **Concentration is therefore worth exactly the gap between this curve and a steeper one**: a router that reached 0.9 with 128 experts would hold a full-attention layer for 0.23 GiB instead of 0.46 GiB. Whether any such router exists at acceptable quality is the experiment; nothing in the data says it does or does not.

**B1.2 The loss.** With W_r the F32 router (2048 × 256 per layer), the trust copy p⁰ of the served router, and the assistant-token mask m_t (7.12):

1. *Soft-mass term* (already in 7.12): L_res = − Σ_l w_l · mean_t log(Σ_{e∈R_l} p_{l,t}(e) + ε).
2. *Zero-miss hinge* (new). The top-8 of z lies inside R_l **iff** the 8th-largest logit inside R_l exceeds the largest logit outside it. Define the margin μ_{l,t} = z_{l,t}^{(8)}[R_l] − max_{e∉R_l} z_{l,t}(e), and
   L_zm = Σ_{l∈T} w_l · mean_t softplus((δ − μ_{l,t})/τ) · τ,
   where T is the target layer set (empty by default; the 10 full-attention layers for L-Z), δ the margin (set from P-E3 as the 25th percentile of the positive margins — DECISION rule) and τ a temperature (δ/4, starting value). The gradient flows to two logits per token-layer (the weakest in-set winner and the strongest out-of-set competitor), which is the minimum change that turns a miss into a hit.
3. *Distillation anchor* (new; this is the "keep the cold experts' knowledge" term). L_KD = truncated KL(q_C0 ‖ q_θ) on a general-text mix, where q_C0 is the stored top-256 + tail distribution of checkpoint C0 (9.6 format; no teacher forward at step time, so the cost is one extra loss kernel on the already-computed logits). It stops the core and router from buying locality by forgetting what the out-of-set experts contributed.
4. *Trust term* (7.12): λ_trust · KL(p ‖ p⁰), all 256-way.

Total: L = L_CE(+MTP) + λ_trust·KL(p‖p⁰) + λ_res·L_res + λ_zm·L_zm + λ_KD·L_KD. Per-layer weights w_l = c_l · (expert bytes share of layer l), with c_l = 1 until P-E2 measures the cost of a miss per layer type (P-E2 may say full-attention layers cost more; then c_l rises there). All λ start at 0 except λ_trust (the control run), and each is added in its own short continuation from C0, so the sweep never contaminates C0 (7.12 Phase 1).

**B1.3 Clamped-forward variant (L-C) — training absorbs the clamp, then the router is taught to need no clamp.** The paper's stacking result (a model whose router was trained for locality pays +0.9 % for a rerouting step that costs +7.7 % on the baseline) says the *rest of the network* can adapt to a restricted expert set. L-C exploits that in two stages, with no inference-time clamp anywhere in the shipped model:

* *Stage A (absorb).* In the trainer's forward, mask the logits of experts outside R_l (to −∞) before the top-8 on the target layers. The tape records the clamped selection and the backward uses it (D7), so the step is self-consistent. The core LoRA learns to produce good outputs from the experts that will be resident. The loss is CE (+ L_KD). This costs nothing extra: it is a mask in the router kernel.
* *Stage B (align).* Remove the mask. Train W_r (core frozen or at a small learning rate) with L_zm + L_res + λ_trust so that the **unclamped** router picks inside R_l on its own. Quality is judged on the unclamped, as-served forward (that is what the engine will run); Stage A's clamped CE is a diagnostic only.

L-C is ablation T-A12 (7.11), run only if the plain joint version (B1.2) fails G-L1 or G-Z2 at two learning rates. The pre-registered reading is the paper's: if Stage B cannot reach the zero-miss gate at ≤ 1 % CE cost, locality is a property the 4-bit-frozen experts cannot give, and the mode is closed.

**B1.4 Choosing R_l.** The resident set is not chosen by the trainer; it is the engine's water-filling on the tape, which already gives each layer its own k_l (84 to 161 experts per layer at 7.5 GiB, READ-SRC). The trainer's job is to reproduce that choice exactly (B4, `PELIAS_HOT_SET` writer) and to feed it the *trained* model's tape (7.12 Phase 2: at most two rounds, stop at mean Jaccard ≥ 0.9). In L-Z the set is chosen by a two-step rule: fix k_l = k_T on the target layers (for example 128 or 192 from the table in B1.1), then water-fill the remaining bytes over the other layers; the budget equation is Σ_l k_l·s_l + Σ_l s_l(zero slot) ≤ 7.5 GiB.

**B1.5 Gates and kill criteria for L-Z (declared before any run).**

| Gate | Pass | Measured by |
|---|---|---|
| G-Z1 | Held-out zero-miss share on the target layers ≥ 0.90 at the chosen k_T (DECISION; the untrained value at k = 128 is 0.52) | tape + engine simulator (P-E1 script, 16.6) |
| G-Z2 | Held-out assistant-token CE ≤ 1.01 × C0 and no regression on the author's generation checks | `eval`, P-L10 |
| G-Z3 | Overall byte-weighted hit ≥ C0's − 0.01, and each domain class ≥ C0's − 0.01 (G-L3) | P-L6 |
| G-Z4 | Engine decode above C0's by more than run-to-run noise (3 launches per side) | P-L7 |

Closed without code if P-E3 shows the median shift needed to reach the margin exceeds what the trust region allows (P-L4), or if P-E2 shows a miss costs the same on every layer type *and* the plain Mode L of 7.12 already passes G-L2 (then target layers add nothing). Closed after trial if no (λ_zm, δ, learning rate) passes G-Z1 and G-Z2 together at two router learning rates.

**B1.6 Expert consolidation (R5) — not built, and why.** Consolidating a layer's experts into k < 256 means training those experts. One layer at k = 128 has 128 × 3 × (2048 × 512) = 402.7 M parameters; for the ten full-attention layers that is 4.03 B parameters, which needs fp32 masters (16.1 GB) and Adam moments (32.2 GB) on top of the streaming design — more than the 48 GiB host cap leaves (DERIVED), and it needs the exact expert quantizer at 4-bit with straight-through gradients (P-E6 is the prerequisite). It is reopened only if L-Z closes *and* P-E2 shows the full-attention layers' misses are dominant in the token latency.

**B1.7 An engine lever this exposes (handed to the engine, not a trainer feature).** The engine's water-filling maximizes the byte-weighted hit rate. The engine's own timeline (ledger §9.2) shows a large *fixed* cost per layer that has any miss (sync-and-copy and the CPU trip) next to a smaller per-expert cost. If P-E2 confirms c₀ ≫ c₁, the right objective is the zero-miss share Z_l, not h; and because the coverage curve above is S-shaped (0.12 at 64, 0.52 at 128, 0.94 at 224), the optimum may be to concentrate bytes in some layers rather than spread them thinly. In the baseline allocation 63.9 % of layer-steps have at least one miss. This is the first item for the engine cross-check once the pause is lifted.

#### B2 — Byte budget and format closure (D22)

Already specified in 9.4; this bridge fixes the inputs. The table in 16.3.1 gives Δ_max = 0.16 GiB (default micro-batch) or none (`-ub 2048`) on the whole-card reading of the 10.7 GiB ceiling, and ≈ 1.2 / 0.9 GiB on the process-only reading (16.7, decision 7). The solver is run for the micro-batch the engine ships with; until then both are reported. Three rules: (1) output types are limited to the engine's supported set (P-E5); (2) a class that gets no affordable raise is closed by Mode Q (4.10) at its served type, which costs zero extra bytes; (3) every engine milestone that changes its non-bank allocation (E7 spine, E8 graphs, E9 planner) re-runs P-E4 and the solver. Exchange rate for judging Mode L against bytes: ≈ 0.027 hit per 0.59 GiB (DERIVED from the engine's static-bank curve).

#### B3 — The accuracy gates, aligned with the engine's

Three different KL measurements, with three different gates; none compares a fine-tune with the base model (they are supposed to differ).

| Gate | Compares | Pass | Where |
|---|---|---|---|
| **K1 parity** | trainer forward vs llama.cpp on the **same** GGUF | truncated KL ≤ 1.0 × σ_float | rung 3 (10.2) |
| **K2 export** | exported GGUF run in llama.cpp vs the trainer's reference logits | ≤ 1.0 × σ_float (adapter-served and merged both reported; their difference is the merge error) | rung 7, 9.6 |
| **K3 engine** | the engine on the exported GGUF vs llama.cpp on the same GGUF | the engine's ruling: float spread governs (H6), initial values 0.017353 (batch 16), 0.018286 (batch 1024) | engine E3/E7/E8, P-X5 |

σ_float is the stock-vs-stock KL between llama.cpp batch configurations. **The engine's two numbers belong to the current served file and its 16 × 1,024-token test set; they are not constants of the architecture.** So the order is: (i) the trainer's harness reproduces the two numbers on the current served GGUF with the engine's protocol (16 chunks × 1,024 tokens, reference stock batch 8) — reruns are deterministic (ledger §9.3), so the pass is equality to the printed precision (P-Q4, step a); (ii) only then is σ_float re-measured on the new GGUF and the trainer's tokens (P-Q4, step b), and K1–K3 use that.

#### B4 — Export alignment, layout manifest, hot-set profile, repack ownership

**Who owns what.** The *standard GGUF* is the interchange format: llama.cpp must load it (the oracle needs that), so nothing in it may depend on the engine's private layout. The *packed layout* and its repacker (`repack_gguf.py`, listed under the engine's own `tools/`) belong to the engine; the trainer neither defines nor re-implements it. The trainer provides (1) the standard GGUF, (2) a layout manifest, (3) the hot-set profile, (4) an optional call into the engine's repacker.

**What the trainer cannot change.** The cold experts are frozen bytes copied verbatim (9.3). Whether they are read in 16-byte chunks is decided by the engine's kernel and layout, not by anything the trainer trains. The trainer's core tensors live in VRAM and are never streamed over PCIe. So alignment is a property of the **file layout and the repack**, not of the training.

**Facts that shape the layout (READ-SRC, P-E9; DERIVED arithmetic).**

| Fact | Value |
|---|---|
| Q4_K block / row (gate, up: 2048 inputs) / row (down: 512 inputs) | 144 B (multiple of 16) / 1,152 B / 288 B — all 16-B multiples |
| Q6_K block / row (down: 512 inputs) | **210 B** (210 mod 16 = 2) / **420 B** (420 mod 16 = 4) |
| Expert tensor sizes | 144 MiB (Q4_K) or 210 MiB (Q6_K) — multiples of 2 MiB |
| Served file | alignment key absent (32); data starts at byte 10,991,104; every expert tensor offset is a multiple of 16; 9 of 123 are 4-KiB-aligned; none 2-MiB-aligned |
| Padding Q6_K rows to 16 B (420 → 432 B) | +2.86 % on those tensors: **120 MiB over the engine's 20 Q6_K layers (+0.65 % of the 18.16 GiB expert bytes)**; the 21st Q6_K tensor is the MTP block, outside the engine |

**Steps added to the exporter (after 9.3 step 6).**

1. *Layout planner (optional, `--layout expert-last`).* Order the tensor infos: all non-expert tensors first (original order), then the 123 expert tensors layer-major. Keep `general.alignment` 32. Add a padding metadata string `pelias.pad` whose length is chosen so that the first expert tensor starts on a 2-MiB boundary; because every expert tensor is a 2-MiB multiple, **all of them** then start on 2-MiB boundaries, at a cost of < 2 MiB of header. The ggml reader requires tensors to be tightly ordered with 32-byte padding only, which this satisfies. *Check:* P-E9 loads the reordered file in llama.cpp and compares logits with the natural layout (must be identical); if the loader refuses, the planner falls back to the natural order and the manifest says so. Row-level padding **cannot** be done in a standard GGUF (the row size of a type is fixed by the format) — that is the pack's job.
2. *Layout manifest* `layout_manifest.json`: for every tensor, name, type, shape, bytes, absolute offset, offset mod 16 / 4 KiB / 2 MiB, bytes per row and per block, blocks per row, SHA-256; per layer, the three expert sizes and the total (this is the content of the engine's `expert_layer_sizes.json`); the file's SHA-256 and the trainer commit. The engine's GGUF reader (E3) can check its parse against it.
3. *Hot-set writer (D25).* Inputs: `expert_usage.json` (tape counts), the manifest sizes, the bank budget (7.5 GiB less one zero slot per layer, 72.6 MiB in all), the engine's 70 / 30 fit / held-out split. Algorithm: the engine's water-filling over expert frequency per byte. Output: the `PELIAS_HOT_SET` text file and its JSON provenance, **byte-identical** to `export_hot_set.py` on the same inputs (P-E14), including the engine's own parser mirror accepting it. Until P-E14 passes, the engine's script is the source of truth and the trainer only calls it.
4. *Optional pack (`--engine-pack`).* Calls the engine's repacker on the GGUF and the manifest. The pack records the GGUF SHA-256; `unpack(pack(x))` must equal the original expert bytes; the pack is never used as oracle input. Which layout (Q6_K row padding, planar split of `ql/qh/scales/d`, anything else) is the engine's choice after P-E10; the trainer waits for a written layout specification before writing any pack code.
5. *Verify.* The existing proofs (9.3 step 6) plus: manifest offsets equal the re-read GGUF's; expert tensor bytes hash-equal to the base's.

What stays optional until measured: 2-MiB expert alignment (P-E11), Q6_K row padding (P-E10).

#### B5 — Dual-branch numerics

**What the two branches really compute (READ-SRC for the arithmetic; the engine's own kernels are not written yet).** In both llama.cpp paths the weight is a K-quant, the activation is quantized to 8 bits, the dot product of the integer codes accumulates in **int32**, and the block scales (`d`, `dmin`, the 6-bit sub-scales, the activation scale) are applied in **fp32**. There is no fp16/bf16 accumulator in these paths, so the premise in R21 is not what separates the branches. What does separate them:

| | CPU branch (llama.cpp AVX2 `vec_dot`) | GPU branch (CUDA `mmvq`) |
|---|---|---|
| Activation quantization | `Q8_K`: one fp32 scale per **256** values | `q8_1`: one scale per **32** values |
| Effect of one outlier in the input | inflates the scale of 255 neighbours | inflates the scale of 31 neighbours |
| Summation order | AVX2 horizontal adds | warp-shuffle reduction |

So an activation spike costs the CPU branch more precision than the GPU branch, and the merged layer output is a sum of the two. How much that matters for KL is unmeasured (P-E12 measures it per token-layer-expert triple against the outlier index ρ = max|x| / rms(x) of each 256-block).

**Router near-ties.** The router runs on the GPU spine in the engine's design and in the tier build; what can differ between llama.cpp and the engine is float noise in z. A selection flips only if the 8th and 9th logits are closer than that noise. P-E3 now records the 8th–9th gap; with the logit noise from P-Q4 it gives the fraction of token-layers at risk. A router-margin regularizer is added only if that fraction is material (> 1 % of token-layers; DECISION starting value).

**D24 — Mode B / N (gated).**

* *Mode B (bimodal-faithful forward).* In the trainer's expert forward, quantize the input of every (token, layer, expert) assignment outside R_l with the CPU's `Q8_K` rule (256-blocks) and those inside R_l with `q8_1` (32-blocks), as the served dual-branch engine will. The LoRA then trains against both noise models in their served proportion. Cost: one small extra quantizer kernel; no extra memory.
* *Mode N (outlier regularizer).* L_out = λ_out · mean over (layer, token, 256-block) of relu(ρ − ρ₀)², on the MoE input u (2048 = 8 blocks) and on the SwiGLU output h (512 = 2 blocks) of the selected experts. ρ₀ is the 95th percentile of ρ on the base model (from P-E12). It acts through the core LoRA and the trainable norm gains.
* *Adopted only if* P-E12 shows Pearson r ≥ 0.3 between ρ and the CPU/GPU expert-output difference **and** the top 5 % of ρ carry ≥ 50 % of the excess KL (P-S6). Otherwise both are closed and nothing is built. Kill criterion after trial: Mode N must not raise CE by more than 0.5 % and must lower K3 on the engine's test set by more than its run-to-run spread.

**Rule recorded for the engine (not a trainer feature).** Integer dot products accumulate in int32 and scales apply in fp32 in every kernel, CPU and GPU; a kernel that accumulates in fp16/bf16 is a bug. Whether to use one activation granularity on both branches (self-consistent) or to match llama.cpp's two (closer to the oracle) is the engine's decision; P-E12 gives the data.

### 16.6 MEASURABLES — what has to be derived, how, how it is used, and where it applies

This is the separate register the author asked for. "Owner" says who runs it. "State" is **done**, **ready** (no code needed, can run in Wave 0) or **needs code**. Section numbers are in this document; "engine" means the engine blueprint.

| ID | Datum to derive | Proposed method (short; full text in 11.9 unless noted) | How it is used | Applies in | Owner / state |
|---|---|---|---|---|---|
| P-E1 | Does the full-attention layer type route differently; what does full residency cost; how fast does all-8-inside coverage grow with set size | Read-only run of the engine's trace simulator (script spec below) | Decides whether layer weights are uniform; gives the coverage curve of B1.1 | 7.12, 16.5 B1, 14.3 | trainer / **done** |
| P-E2 | Cost of a layer as a function of its miss count, by layer type and context depth | Engine's per-layer timers (`e0_timeline.patch`); regress latency on misses with layer type as a factor | c_l in the loss weights; whether the engine should optimise zero-miss share instead of byte-hit (B1.7) | 7.12, B1.2, B1.7, engine E5/E12 | engine / ready |
| P-E3 | Logit margins between in-set and out-of-set experts; 8th–9th gap | Router logits via the P-S1 callback over the trace corpus; sets R_l of 96 … 224 | δ of the hinge; feasibility of L-Z; near-tie flip rate | B1.2, B1.5, B5 | trainer / ready |
| P-E4 | VRAM left for the core, by bank, prefill micro-batch and context | Engine reference cell with byte-only core variants (`llama-quantize --tensor-type`); NVML peak | Δ_max for the solver | 9.4, B2, 16.3.1, 1.6 | engine / ready |
| P-E5 | Weight types the engine will support | Read the blueprint (E4/E5/E7), ask the author | Output type set of D22 | 9.4, B2 | author / ready |
| P-E6 | Bit-exactness of our ported quantizers | CPU port of `quantize_row_*_ref` vs ggml on random, outlier-heavy and real blocks, then GPU vs CPU | Gate for Mode Q and the exporter | 4.10, 9.3, B2 | trainer / needs code (small) |
| P-E7 | Do the served base weights sit exactly on their grid | Fraction of blocks with Q(deq(block)) = block, per class and type | Validates the erasure model of 4.3 | 4.3, 9.4 | trainer / needs P-E6 |
| P-E8 | Does Mode Q close the merge gap, and at what step cost | Slice, then full model; post-export CE through llama.cpp | Adopt Mode Q per class | 4.10, 9.4 | trainer / needs trainer |
| P-E9 | Alignment of tensors, rows and blocks; does the expert-last layout load identically | Header-only read; load the reordered file in llama.cpp, compare logits | Layout planner choice; manifest | B4, 9.3 | trainer / header part **done** |
| P-E10 | Does a 16-B, shared-memory-staged kernel reach the checksum rate; does Q6_K need padding | Engine `wide_load.cu` variants, Q6_K rows raw vs 432 B | Pack layout; manifest flags | B4, engine E5 | engine / ready |
| P-E11 | Does 2-MiB backing change zero-copy read speed (WSL2, Windows) | 4 KiB vs 2 MiB-backed pinned buffers; read `AnonHugePages` per load mode | Keep 2-MiB alignment optional or require it | B4, 16.3 X3, engine unknowns | engine / ready |
| P-E12 | Does CPU/GPU expert-output difference grow with activation outliers | P-S1 dumps; per-block ρ vs relative difference | Mode B/N adopt or close | B5 | trainer / ready |
| P-E13 | Accumulator types and reduction trees in the engine's and llama.cpp's kernels | Static audit + injected-outlier test against fp64 | The int32/fp32 rule for the engine | B5 | engine / when kernels exist |
| P-E14 | Does the C++ hot-set writer match `export_hot_set.py` byte for byte | Same inputs through both; run the engine's parser | Trainer may own the writer | B4 | trainer / needs code (small) |
| P-Q3 | Which bit-widths and types preserve the learned delta, per class | Merge sweep on the v0.2 adapter | Type per class | 4.3, 9.4 | trainer / ready |
| P-Q4 | σ_float: the stock-vs-stock KL spread; reproduction of the engine's two numbers | Engine protocol on the current served GGUF (step a), then on the new GGUF and trainer tokens (step b) | Every KL gate | B3, 10.3 | trainer / ready |
| P-S1/S6 | Per-layer reference dumps; where the served KL excess sits | `eval-callback` dumps on fixed sequences | Fixtures for rungs 2–3, P-E3, P-E12 | 10.2 | trainer / ready |
| P-L1/L2/L9 | Static-vs-Belady gap; cost of an untrained clamp on this model; router-LoRA support | 11.8 | Size of the Mode L prize | 7.12, B1 | trainer / ready |
| P-H11 | Faster platform for training: WSL2 vs Windows-native | Same bench on both | Platform default; the engine saw +25–28 % decode on Windows-native | 5.1, 15.7, 16.3 X9 | trainer / ready |
| P-X5 | Engine regression with the real merged core | Reference cell, 3 launches per side | Accept or reject an export | 9.5 | engine+trainer / after export |

**Engine-owned measurements the outside report suggested (no trainer dependency; recorded so none is lost).** E-A1 thread count and pinning, 8 physical vs 16 SMT (ledger shows 5–7 beat 10 on stock); E-A2 acquire/release vs seq_cst on handoff flags; E-A3 `PREFETCHNTA` on cold reads; E-A4 L2 persisting window (query `cudaDevAttrMaxPersistingL2CacheSize` first); E-A5 `AnonHugePages` per load mode (the engine's open unknown); E-A6 native-Linux floors (the engine's E1); E-A7 pinned memory above 20 GiB. Each is an A/B in the engine phase named in 16.7, with its own go/no-go.

**P-E1 script specification (method of the run already done; the code is not part of this document).**

1. Load the engine's merged decode routing trace (121,160 decode steps × 40 layers × 8 experts) and the per-layer expert byte sizes; take its 70 / 30 fit / held-out split.
2. Per-layer expert frequency from the fit split; the engine's water-filling selection at 7.5 GiB less the zero slots → baseline resident mask. *Check:* held-out byte-weighted hit 0.8273 and any-miss share 0.639, 4,170 experts — both reproduced exactly.
3. For the held-out steps: per-layer hit, per-layer zero-miss share (all 8 inside), the histogram of misses per layer-step, means over the 10 full-attention layers (indices 3, 7, …, 39) against the other 30.
4. Scenario B: mark the ten full-attention layers fully resident (4.605 GiB), water-fill the rest with the bytes left, report the same.
5. Coverage curve: rank each layer's experts by fit frequency; a held-out step is all-8-inside for set size s iff its worst-ranked expert has rank < s; report the share for s = 64 … 256 and the s needed for 90 / 95 / 99 %.

Results (DERIVED, 2026-10-23): baseline hit 0.8273, any-miss 0.639, miss histogram 0.361 / 0.279 / 0.165 / 0.093 / 0.051 / 0.028 / 0.014 / 0.007 / 0.003 for 0…8 misses; full-attention layers hit 0.8299, others 0.8265; zero-miss 0.3636 vs 0.3602; baseline gives the full-attention layers 1,045 experts (1.878 GiB) and the others 3,125 (5.55 GiB). Scenario B: full-attention layers 1.000, others 0.6164, overall 0.7114, zero-miss 0.3197, any-miss 0.6803. Coverage curve and s needed: table in B1.1. The script runs in seconds on the engine's own data and is re-run on every Mode L candidate.

### 16.7 Changes proposed for the engine side (not yet made), the two-way contract, and open decisions

**Edits proposed for the engine blueprint — not yet made; they are the first item after the pause the author asked for.** (1) Scope line: the description of arXiv 2608.18261 corrected, and "router changes" split into inference-time (out of scope) and trainer-side (an input). (2) A VRAM-budget table: ceiling (10.7 for the trainer; the engine blueprint still says 10.9 and its gates 11.2; 16.3 X15), 7.5 GiB bank, peaks at both micro-batches, the headroom that remains for the core. (3) The CPU-read row of the inputs table corrected (CPU alone 43.8–45.9; 26.6 is the share under contention). (4) E3 gate: the oracle also reads trainer-exported GGUFs and reproduces the noise floor on them. (5) E5: slot table supports per-layer capacities and a hot-set file from the trainer; the wide-load kernel must handle Q6_K's 210-byte blocks (stream-aligned staging). (6) E11: the startup probe also records the transparent-huge-page setting and `AnonHugePages` during load. (7) E12: hot-set objective (byte-weighted hit vs zero-miss share) decided after P-E2; trainer-side Mode L candidates enter the extras list. (8) The unknowns table gets the outside report's hypotheses as separate A/B items. (9) A new section, "Trainer co-design contract", with the table below. (10) Sources: the ledger sections used and the outside report as an unverified input.

**Two-way contract.**

| Engine milestone | Releases to the trainer | Trainer output the engine checks |
|---|---|---|
| E0 per-layer timers | P-E2 → layer weights c_l (B1.2) | — |
| E1/E11 platform results | Platform default for training (P-H11) | — |
| E3 reader and oracle | Reproduction of the stock KL floor (B3) | Layout manifest and exported GGUF parse identically; K3 gate holds on the exported GGUF |
| E5 bank and wide-load kernel | P-E10, P-E11 → manifest flags, pack decision | Hot-set file accepted by the slot-table loader; parity P-E14 |
| E7 full spine | P-E5 type matrix, P-E13 audit | Core types within the supported set |
| E8/E9 decode loop, VRAM planner | Δ_max (P-E4) at the shipped micro-batch | Peak VRAM at the 7.5 GiB bank ≤ 10.7 GiB with the real merged core |
| E12 extras | LRU only if live hit rate is well below simulation; MTP | Mode L candidates: re-profiled hot sets, measured decode vs C0 (G-L4, G-Z4) |

**Decisions for the author (the trainer is written so that none of them blocks Wave 0).**

1. Is the 7.5 GiB bank a ruling, or the current best step under the ceiling (10.7 for the trainer)? (This document treats it as derived.)
2. Which prefill micro-batch does the engine ship with? It moves Δ_max between 0.16 GiB and none on the whole-card reading (it was 0.36 and ≈ 0.11 GiB at 10.9), and so decides which core classes can be raised above their served type.
3. Does the engine add `Q8_0` weights? (Only then is Q8_0 an output type.)
4. Is Windows-native promoted from "not in scope" to a supported target? The recorded runs (two Windows launches, one WSL launch) show +25–28 % decode; three launches per side are needed before it is a claim.
5. Is Mode L / L-Z worth building, once P-L1, P-L2, P-E2 and P-E3 (all Wave 0, no trainer code) have reported?
6. If the author has a Windows-native run newer than those in the repository (the outside report quoted 57.0 tok/s), it should be added to the ledger with its raw file; until then the recorded numbers are the ones used.
7. **What is the 10.7 GiB ceiling a ceiling of?** **RULED by the author on 2026-10-24:** the ceiling should come from the probe results, so that the system knows the best number for the memory available (3.3, U2; D28). The author's words were about reading the probe results; the process-only reading (P) is the one a probe can measure, so P is recorded as the governing reading (the author may correct this). The question as it was asked: (W) the whole card as NVML sees it, Windows' share included (how the engine's peaks, its 11.2 GiB gate and the runtime check of 15.1 are measured), which leaves the core 0.16 GiB at the default micro-batch and none at `-ub 2048`; or (P) the process's own allocations after the context exists (how P-H8, the plan sum and `cudaMemGetInfo` read it), which leaves about 1.2 / 0.9 GiB and lets the engine's bank be re-examined. The measured facts: CUDA free after the context is 10.83 / 10.85 GiB; Windows holds 0.89 / 1.01 GiB before the process starts. **Follow-on proposed with the ruling (not yet agreed):** 10.7 GiB stays this machine's value, and the governor's start rule is unchanged, until the margin is measured (11.12); chapters 4, 5.3, 9.4 and 12 keep the whole-card figures as the stricter planning figures (lower bounds); the P figures in 16.3.1 become the working headroom once P-E4 has re-measured the engine's non-bank footprint with NVML used recorded before the engine starts.

---

## Appendix A — Tensor name map (READ-SRC: the served file's header, read 2026-10-23)

753 tensors; `{N}` is the block index. 41 blocks: 0–39 are the trunk, 40 is the multi-token-prediction (MTP) block. GGUF shapes list the inner (contiguous) dimension first. The model-file name is not printed here by design. Hugging Face–side names are not asserted in this document; the conventions probe (P-S2, P-S3) derives the mapping and the inverse permutations on the oracle side.

| GGUF name | Type(s) in the file | Shape | Count | Role | In the trained core? |
|---|---|---|---|---|---|
| `token_embd.weight` | Q4_K | 2048 × 248,320 | 1 | input embedding | no (frozen; byte-copied) |
| `output.weight` | Q6_K | 2048 × 248,320 | 1 | lm_head (untied) | no; optional LoRA (T-A8) |
| `output_norm.weight` | F32 | 2048 | 1 | final RMSNorm | optional (norms) |
| `blk.{N}.attn_norm.weight` | F32 | 2048 | 41 | pre-attention / pre-DeltaNet norm | optional |
| `blk.{N}.post_attention_norm.weight` | F32 | 2048 | 41 | pre-MoE norm | optional |
| `blk.{N}.attn_qkv.weight` | Q6_K ×14, Q4_K ×16 | 2048 × 8192 | 30 | DeltaNet joint q, k, v projection | **LoRA** |
| `blk.{N}.attn_gate.weight` | Q4_K | 2048 × 4096 | 30 | DeltaNet z (output-gate) projection | **LoRA** |
| `blk.{N}.ssm_alpha.weight`, `ssm_beta.weight` | Q4_K | 2048 × 32 each | 30 + 30 | DeltaNet decay / write-strength projections | **LoRA** |
| `blk.{N}.ssm_out.weight` | Q4_K | 4096 × 2048 | 30 | DeltaNet output projection | **LoRA** |
| `blk.{N}.ssm_conv1d.weight` | F32 | 4 × 8192 | 30 | DeltaNet short convolution | frozen |
| `blk.{N}.ssm_a`, `ssm_dt.bias` | F32 | 32 each | 30 + 30 | DeltaNet decay parameters | frozen |
| `blk.{N}.ssm_norm.weight` | F32 | 128 | 30 | DeltaNet gated norm | frozen |
| `blk.{N}.attn_q.weight` | Q4_K | 2048 × 8192 | 11 | attention query **and** output gate, joint | **LoRA** |
| `blk.{N}.attn_k.weight` | Q4_K | 2048 × 512 | 11 | attention key | **LoRA** |
| `blk.{N}.attn_v.weight` | Q6_K ×7, Q4_K ×4 | 2048 × 512 | 11 | attention value | **LoRA** |
| `blk.{N}.attn_output.weight` | Q4_K | 4096 × 2048 | 11 | attention output | **LoRA** |
| `blk.{N}.attn_q_norm.weight`, `attn_k_norm.weight` | F32 | 256 each | 11 + 11 | per-head q/k norms | frozen |
| `blk.{N}.ffn_gate_inp.weight` | F32 | 2048 × 256 | 41 | **router** | frozen; trainable only in Mode L |
| `blk.{N}.ffn_gate_exps.weight`, `ffn_up_exps.weight` | Q4_K | 2048 × 512 × 256 | 41 + 41 | routed experts, gate and up (separate) | no (frozen; byte-copied) |
| `blk.{N}.ffn_down_exps.weight` | Q6_K ×21, Q4_K ×20 | 512 × 2048 × 256 | 41 | routed experts, down | no (frozen; byte-copied) |
| `blk.{N}.ffn_gate_shexp.weight`, `ffn_up_shexp.weight` | Q4_K | 2048 × 512 each | 41 + 41 | shared expert, gate and up | **LoRA** |
| `blk.{N}.ffn_down_shexp.weight` | Q6_K ×21, Q4_K ×20 | 512 × 2048 | 41 | shared expert, down | **LoRA** |
| `blk.{N}.ffn_gate_inp_shexp.weight` | F32 | 2048 | 41 | shared-expert scalar gate | optional (gates) |
| `blk.40.nextn.eh_proj.weight` | Q4_K | 4096 × 2048 | 1 | MTP input fusion | optional (D12) |
| `blk.40.nextn.enorm.weight`, `hnorm.weight`, `shared_head_norm.weight` | F32 | 2048 each | 1 each | MTP norms | frozen |

**Checks.** 123 expert tensors = 41 × 3 (the reader asserts exactly this and tracks the MTP block's three separately — the "82 vs 80 tensors" trap of reviewer claim 9). The v0.2 LoRA module count: 30 × 3 (`attn_qkv`, `attn_gate`, `ssm_out`) + 10 × 4 (attention) + 40 × 3 (shared experts) + 30 × 2 (`ssm_alpha`, `ssm_beta`) = **310** (DERIVED; equals v0.2's `EXPECTED_LORA_MODULES`). Core parameters: 1,409.0 M (the class table of 9.4). `ffn_gate_inp` is 41 × 2048 × 256 = 21,495,808 parameters (82.0 MiB in F32).

## Appendix B — Sources and citations

**Papers (as cited in the text; PUBLISHED unless marked).**

| Reference | Used for |
|---|---|
| arXiv 2608.18261, *Cacheable by Design?* (read in full this session) | routing-locality limits; the stacking result (7.12, 16.3 X2) |
| arXiv 2605.27081, ReMoE (read) | router-only fine-tune with frozen experts (7.12) |
| arXiv 2609.04895, *Cache-Aware Joint Router Adaptation* (abstract read) | listed, not relied on |
| arXiv 2607.08780, *Sticky Routing* (unreadable) | **UNVERIFIED**, not relied on |
| arXiv 2412.06464, Gated Delta Networks; arXiv 2406.06484, parallelizing the delta rule | DeltaNet chunked algorithm (6.7) |
| arXiv 2407.01906, ESFT | frozen-expert default (2.8) |
| arXiv 2411.09009, Cut Cross-Entropy | fused chunked loss (6.8) |
| arXiv 2312.03732, rsLoRA | α/√r option (7.2) |
| arXiv 2408.11743, MARLIN | dequant-in-register GEMM shape (6.2) |
| arXiv 2601.14277, llama.cpp quantization study | magnitude of Q4_K/Q6_K/Q8_0 loss (4.3) |

**Documents and code.**

| Reference | Used for |
|---|---|
| pelias-engine blueprint 2026-10-22; results ledger `benchmarks/RESULTS.md` sections 1–9.4; `raw/e0-2026-10-22/` (incl. machine state); `raw/windows-native-2026-10-23/` | Chapters 2.7, 9, 16 |
| llama.cpp commit `2ca15f540` (b11399), MIT: `ggml-quants.c` reference quantizers, `eval-callback`, `llama-perplexity`, `llama-quantize`, the qwen35moe graph | the oracle and the quantizer ports (P-E6) |
| GGUF specification and `gguf-py` | file layout, alignment (9.3, B4) |
| NVIDIA, *CUDA on WSL User Guide* (13.0.3 archive) | WSL2 limits (engine blueprint; 15.7) |
| Microsoft Learn, *GPU virtual address* (WDDM), section "Physical page size" — fetched 2026-10-23 | 16.4 G2 |
| AMD, *Software Optimization Guide for Family 19h* (pub. 56665, rev. 3.00), §2.14 Table 3 and §2.15 — fetched 2026-10-23 | 16.4 G6, G7 |
| Higham, *Accuracy and Stability of Numerical Algorithms*, §3.1 | error bounds used in the engine's cold-path tests (cited by the ledger) |
| FlashAttention-2 (BSD-3), flash-linear-attention (MIT), KTransformers (Apache-2.0), Unsloth (AGPL — nothing copied) | prior art (2.8), licensing (5.1) |
| An outside AI-generated "tri-environment hardware-verified research plan", 2026-10-23 | **input to be checked**, not a source; 16.4 |

## Appendix C — The 13 reviewer claims as a trap checklist for a native trainer

The reviewers' 13 audited bugs (second reviewer file, section I.A) concern PyTorch / `transformers 5.17` / Triton internals. None applies line by line to a native trainer, and none was re-verified here. Each is kept as a **trap to test the new code against**.

| # | Reviewer's trap (UNVERIFIED here) | Native-trainer analogue | Where it is addressed / tested |
|---|---|---|---|
| 1 | Trainable gates and norms: dtype mismatch on step 1; weight decay pulls zero-centered norm scales toward −1 | Norm/gate gradients in fp32; weight decay is 0 on norms, gates, biases and magnitudes | 6.8 (O1), 7.1; P-T6; P-T1 includes a gradient test on a trainable norm |
| 2 | Checkpointing hook is a no-op; the first layer's input does not require grad, so the autograd graph is dropped | No autograd exists; boundaries and recompute are explicit | 5.6; P-T1, P-T5 (the loss is independent of how the step is split) |
| 3 | `is_grad_enabled()` is always false inside a custom forward, so the MoE output is never stashed | The routing tape is written in the training forward only and read (not recomputed) in the backward | D7; 7.4; P-T1, P-T4 |
| 4 | Layers recomputed twice; pageable `.cpu()` syncs; mask API removed | Each layer is recomputed exactly once per backward (kernel-launch counter in the test); all boundary traffic is pinned; padded rows use document boundaries, never a materialized mask | 5.4–5.6; P-D3, P-T7, P-H12 |
| 5 | `auto` backend silently picks the slow path; 120 host syncs | Kernel choice is static and logged; routing plan computed once and reused by the backward | 6.3; P-K4, P-K6 |
| 6 | Tile-map kernel runs a 256-iteration loop on one thread | The expert-grouping/sort kernel is block-parallel and benchmarked | 6.3; P-K4 |
| 7 | A dense 512 MiB 4-D causal mask at 16 k tokens | Attention uses implicit causal and document-boundary masks; no [T, T] buffer exists in the memory plan | 5.3, 6.6 |
| 8 | 3-D vs 2-D hidden-state indexing | Fixed-shape structs of pointers; shape asserts in debug builds; B > 1 unit test | 5.1, 6.0 |
| 9 | Loading the MTP block breaks the "2 × L expert tensors" assertion | The reader asserts 123 expert tensors (41 × 3) and tracks the MTP block's separately | Appendix A; 7.6; P-S5 |
| 10 | Activation-importance calibration order (needs a loaded model before the quantized cache) | No imatrix: the served file was quantized without one, so re-quantization must be imatrix-free to be idempotent | D9; P-E7 |
| 11 | Multi-block think-span scanner and tiered span weights | Label masks come from the served chat template; weights are an ablation (uniform vs tiered) | 8.4; T-A6; P-D5 |
| 12 | Probe lacks vendor sampling settings | Generation checks run in llama-server with the vendor settings; the trainer needs only teacher-forced CE | 7.7; engine speed protocol (temperature 1.0, top-p 0.95, top-k 20) |
| 13 | Mismatched flags, hard-coded bounds, a pad token that equals the end-of-turn token | All options are declared once in a single options header; packed pad slots have weight 0 and are never attended; the pad id is never `<|im_end|>` | 5.1, 8.5; P-D3 |

## Appendix D — Decision register

Status: **kept** · **revised** (changed by this audit) · **new** (added by this audit). Numbers D11 and D15–D20 were used in earlier drafts for items that were merged or dropped; no text refers to them.

| ID | Decision | Where | Status |
|---|---|---|---|
| D1 | GGUF in, GGUF out: train against the bytes the engine serves | 1.2, 2 | kept |
| D2 | The routed experts are frozen and byte-spliced into the output (the router is the one gated exception, D21) | 1.2, 9.3 | kept (amended) |
| D3 | The trained part is a LoRA on the 310 core modules, rank 16, fp32 masters | 7.2 | kept |
| D4 | The core is exported as an isolated higher-precision block; **types come from the byte-budget solver, not a fixed Q8_0** | 4.3–4.4, 9.4 | **revised** |
| D5 | Layer-major scheduling with K micro-batches per layer sweep | 5.6 | kept |
| D6 | Two forward numerics (bf16 reference, serve-faithful int8) chosen by measurement | 4.5 | kept |
| D7 | Routing tape recorded in the forward and reused in the backward; router frozen by default | 6.3, 7.4 | kept (amended by D21) |
| D8 | Fused chunked cross-entropy, step-wide token normalization | 7.3 | kept |
| D9 | The base is taken as given (stock `llama-quantize`, no imatrix) | 3.2 | kept |
| D10 | Packing into windows ≤ 8,192 slots; longer documents are a hard failure | 8.5 | kept |
| D12 | MTP co-training conditional on P-S5 | 7.6 | kept |
| D13 | Three streams (compute, weight loads, boundary traffic) | 5.4 | kept |
| D14 | Coding basis: C++20 / CUDA 13.0.88, `sm_89`, CMake + Ninja, no Python in the product | 5.1 | kept |
| D21 | Mode L — locality-aware router training; optional, gated, off by default. **Amended** by 16.5 B1: zero-miss hinge (L-Z), clamped-forward absorb (L-C), distillation anchor, gates G-Z1…G-Z4 | 7.12, 16.5 | new (amended) |
| D22 | Format closure and byte-budget solver; Δ_max = 0.16 GiB (default micro-batch) or none (`-ub 2048`) at the 10.7 GiB ceiling, whole-card basis (it was 0.36 / ≈ 0.11 at 10.9) | 9.4, 16.3.1 | new (numbers revised) |
| D23 | Mode Q — merge-aware training through the exact ggml quantizer (straight-through) | 4.10 | new |
| D24 | Mode B / N — bimodal-faithful forward and outlier regularizer; adopted only if P-E12 passes | 16.5 B5 | new |
| D25 | Engine contract: hot-set writer (parity P-E14), layout manifest, optional expert-last layout, optional call into the engine's repacker; the engine owns the packed layout | 16.5 B4, 16.7 | new |
| D26 | VRAM ceiling 10.7 GiB (author ruling 2026-10-23, from 10.9), judged at start on the smaller of CUDA free and NVML free after the context; its basis (whole card vs process) is open (16.7 decision 7) | 0.3, 15.1, 16.3 X15, X16 | new |
| D27 | The streamer flushes (`cudaStreamQuery`) after each slot load, boundary copy and kernel batch, once per multi-millisecond batch; the solver assumes overlap efficiency 0.84 (pair) and 0.92 (all three) | 5.4, 11.12 (P-H12, P-H9) | new |
| D28 | The VRAM ceiling is measured at start from the probes (smaller of CUDA free and NVML free after the context, minus a margin), the process-only reading; 10.7 GiB stays this machine's value until the margin is measured by the disturbance test | 3.3 (U2), 11.12, 15.1, 16.3.1, 16.7 (decision 7) | author ruling 2026-10-24 |
| D29 | Universal use: a start-up self-probe and machine profile (U1), a plan card with a real choice (U3), a model key from the file's SHA-256 with per-model facts stored under it (U4); built after the top-tier gate | 3.3 | author's points 2026-10-24 |

## Appendix E — Arithmetic ledger

Every derived number in this document comes from one of these computations (all run by script; inputs are labelled). Units: GiB = 2³⁰ B, MiB = 2²⁰ B.

| Quantity | Inputs | Result |
|---|---|---|
| Core size by type | 1,409.0 M parameters; type bits per weight (Q4_K 4.5, Q6_K 6.5625, Q8_0 8.5) applied per class (9.4 table) | served mix 820.3 MiB; all-Q6_K 1,102.3 (+282.0); all-Q8_0 1,427.7 (+607.4); bf16 2,687.5 |
| LoRA parameters and states | rank r over the 310 modules | r = 16: 21.17 M parameters, 323 MiB (fp32 master + grad + m + v); r = 8: 161 MiB; r = 32: 646 MiB |
| FLOP per token | per-op counts of 6.10 | forward 6.74, train 20.39 GFLOP at T = 8,192 (18.13 at 2,048; 18.88 at 4,096) |
| Peak rates (RTX 4070) | published 4090 rows scaled to 46 SMs and the card's clocks (stock); P-H3 sustained rates (overclock) | stock: bf16 58.3 TFLOPS, int8 233.2 TOPS (DERIVED); overclock: bf16 65.2, int8 259.5 (MEASURED; the earlier derived 65.4 / 261.4 were 0.3 % / 0.7 % high) |
| Expert stream | 19,503,513,600 B (18.164 GiB) read twice per step at 26.1 GB/s | 1.49 s per optimizer step, independent of tokens per step; hidden iff tokens per step ≥ 1,495 / 2,242 / 2,989 at 1,000 / 1,500 / 2,000 tok/s compute |
| v0.2 baseline | 616 steps, 202.7 min, median 247.85 tok/s | ≈ 3.01 M processed tokens |
| Router costs (Mode L) | 41 × 2048 × 256 parameters | 82.0 MiB per copy; +0.32 GiB with gradient, Adam and the frozen reference; weight-gradient 41.9 MFLOP per token = 0.206 % of 20.39 GFLOP |
| Hit-rate exchange rate | static-bank curve (GiB: hit) 2.0: 0.43, 3.0: 0.54, 4.0: 0.63, 5.0: 0.70, 5.6: 0.74, 6.0: 0.76, 8.0: 0.85; bank sweep slope | −0.027 hit for −0.59 GiB (7.0 → 6.41); ≈ +1 tok/s per +0.01 hit |
| VRAM headroom at the 7.5 GiB bank | ceiling 10.7; peaks 10.40–10.44 (default micro-batch), 10.67–10.69 (`-ub 2048`); margin 0.1 | whole-card reading: 0.16 GiB and none, −0.09 (16.3.1); at ceiling 10.9 it was 0.36 and ≈ 0.11 |
| Class raises to Q6_K (9.4) | class sizes above | shared experts +25.7, attention +65.5, DeltaNet out +61.9, DeltaNet in +127.8, α/β +1.0 MiB (sum +282.0) |
| P-E1 trace analysis | engine routing: 84,503 fit + 36,657 held-out decode steps × 40 layers × 8 | baseline hit 0.8273 / any-miss 0.639 reproduced; full-attention layers 0.8299 vs 0.8265; full residency 4.605 GiB → overall 0.7114; coverage curve and s for 90/95/99 % = 210 / 227 / 247 (16.5 B1.1, 16.6) |
| Full-attention layer bytes | 4.605 GiB = 4,715 MiB over 10 layers = 4 × 432 + 6 × 498 MiB (432 MiB for Q4_K-down layers, 498 MiB for Q6_K-down) | at set size k: 1,179 / 2,358 / 3,536 / 4,715 MiB for k = 64 / 128 / 192 / 256 |
| Zero slots | one slot per layer, 20 layers × 1.6875 + 20 × 1.945 MiB | 72.6 MiB |
| Q6_K geometry | block 128 + 64 + 16 + 2 = 210 B; row = 2 blocks = 420 B; padded row 432 B | +2.86 % per row; 6.0 MiB per layer; 120 MiB over the engine's 20 Q6_K layers = 0.65 % of 18.164 GiB |
| Consolidation memory (B1.6) | 128 experts × 3 × 2048 × 512 per layer, 10 layers | 402.7 M per layer, 4.03 B total; fp32 masters 16.1 GB; Adam moments 32.2 GB |
| Gap to the ceiling (16.4.1) | ceiling 10.87 / 11.34 ms; runs 37.9, 42.1, 47.3, 46.8, 53.7, 54.1 tok/s | WSL 12.4–15.5 ms; Windows-native 7.2–10.5 ms; Windows–WSL difference 5.1–5.2 ms |
| Headroom, process-only reading (16.3.1) | peaks 10.40–10.44 and 10.67–10.69 minus ≈ 1.0 GiB held by Windows before the process starts (0.89 / 1.01 measured); ceiling 10.7; margin 0.1 | engine's own use ≈ 9.4 / 9.7; headroom ≈ 1.3 / 1.0 before the margin, ≈ 1.2 / 0.9 after |
| Bank given up for an all-Q6_K core (16.3.1) | shortfall 0.28 − 0.16 = 0.12 GiB (default) and 0.28 + 0.09 = 0.37 GiB (`-ub 2048`) × (0.027 hit / 0.59 GiB) | ≈ −0.005 and ≈ −0.017 hit ≈ −0.5 and −1.7 tok/s at ≈ 1 tok/s per 0.01 hit |
| Streamer flush cost (5.4) | P-H9 busy-stream flush 8.05 µs (B) / 76.6 µs (A); a 498 MiB slot load = 522,190,848 B at 26.3 GB/s | 19.9 ms per slot load; one flush per load is 0.04 % (B) / 0.4 % (A) of the load time |
| Launch cost per step (6.9) | ≈ 7,200 launches × 2.18–4.70 µs (P-H9 eager) | 16–34 ms per step, 0.5–1.0 % of 3.3 s |
| Tensor-rate ratios (11.12) | 259.5 / 65.2 and 129.8 / 65.2 (P-H3) | int8 3.98×, fp8 1.99× bf16 |
| Pinned memory per pinned GiB (11.12) | fall of host available memory ÷ pinned GiB (P-H7) | 1.007–1.01 |
