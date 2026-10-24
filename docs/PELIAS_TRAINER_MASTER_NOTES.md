# PELIAS TRAINER — MASTER NOTES

**How each result behind the trainer design was derived, what depends on what, and what is still needed.**
*Companion to [`PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`](PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md) (the "master"). This file does not repeat the master. It records the reasoning trail, so a later reader can check a number without redoing the work.*

| | |
|---|---|
| Date | 2026-10-23; hardware results and the 10.7 GiB ceiling folded in 2026-10-24; universal-use design points and the ruling on decision 7 recorded the same day |
| Status | **Design, plus the Wave 0 hardware measurements. No trainer code exists.** The hardware probes were run on the GPU on 2026-10-23/24 (WSL2 and Windows-native); section 1.18 records what they found and master 11.12 holds the table. The other numbers come from reading files, reading source, running the engine project's logged results through arithmetic, and one analysis of the engine's routing trace (P-E1). |
| Draft | Written quickly from the master. Section 3 (what is needed) is the part to act on. |
| Copyright | Copyright (c) 2026 John Lipka and the Pelias contributors |

**Labels** (same as the master, section 0.2): **MEASURED** (run on the author's machine, in a ledger or raw file) · **READ-SRC** (read from source or from the served model file's header) · **PUBLISHED** (external, cited in master Appendix B) · **DERIVED** (arithmetic from labelled inputs) · **ESTIMATE** (judgement, paired with a probe) · **UNVERIFIED** (not relied on until its probe passes) · **DECISION** (a design choice with a reason).

---

## 1. Discoveries that had little or no prior write-up, and how each was derived

Each entry: **finding** · **how it was derived** · **label** · **what depends on it** · **where in the master**.

### 1.1 What the served model file really contains
* **Finding.** 753 tensors, 20.22 GiB (21,713,463,296 B). 40 trunk layers plus one extra "MTP" block (`blk.40`). Routed experts: gate and up are `Q4_K` in all 41 blocks; down is `Q6_K` in 21 blocks (210 MiB each) and `Q4_K` in 20 (144 MiB each). `lm_head` is `Q6_K`, the embedding `Q4_K`, the routers (`ffn_gate_inp`, 41 × 2048 × 256 = 21,495,808 parameters) are `F32`.
* **Derived by.** Reading the GGUF header with the pinned llama.cpp's own Python reader and tabulating tensor name, shape and type by script. Nothing taken from a model card.
* **Label.** READ-SRC. **Depends on it:** every size in the master; the core's byte budget; the export's byte-splice. **Master:** 2.2, 2.3, Appendix A.

### 1.2 Which layers are which
* **Finding.** 30 DeltaNet (linear-attention) layers and 10 full-attention layers, the latter at indices 3, 7, …, 39. 256 experts, top-8, expert width 512, one shared expert, vocabulary 248,320.
* **Derived by.** Reading the served graph in llama.cpp (`src/models/qwen35moe.cpp`): the rule `(layer + 1) % 4 != 0` selects DeltaNet. Checked against the tensor names that exist per layer.
* **Label.** READ-SRC. **Depends on it:** the module count (1.3), the full-attention residency question (1.9). **Master:** 2.4.

### 1.3 How many things the trainer actually trains
* **Finding.** The trainable set is the "core": shared experts, DeltaNet projections and attention projections, 1,409.0 M parameters. LoRA sits on 310 modules (30 × 3 + 10 × 4 + 40 × 3 + 30 × 2). At rank 16 that is 21.17 M parameters and 323 MiB including fp32 master, gradient and both Adam moments.
* **Derived by.** Summing parameter counts per tensor class from 1.1, then counting modules per layer type. Rank-8 and rank-32 sizes follow linearly (161 and 646 MiB).
* **Label.** DERIVED. **Depends on it:** the memory plan, the optimizer state, the merge cost. **Master:** 4.1, 7, Appendix E.

### 1.4 The core cannot stay at 4-bit once trained (the "delta versus step" problem)
* **Finding.** A small learned change added to a weight stored at 4 bits is often smaller than one quantization step, so re-quantizing the merged weight at the same type can erase it. The core therefore needs more bits than the served mix, or has to be trained through the exact quantizer (Mode Q), or kept as an unmerged adapter.
* **Derived by.** A quantitative argument in master 4.3 (step size of the format against the size of a LoRA update) and a plan to measure it (P-E7: do the served weights sit exactly on their grid; P-E8: does Mode Q close the merge gap).
* **Label.** DERIVED for the argument; the size of the effect on this model is **UNVERIFIED** until P-E7 and P-E8 run. **Depends on it:** the export choice (1.5), D3 and D4. **Master:** 4.3, 4.10.

### 1.5 The core's size at each candidate type, and the 8-bit core did not survive
* **Finding.** Core size: served mix 820.3 MiB; all `Q6_K` 1,102.3 MiB (+282.0); all `Q8_0` 1,427.7 MiB (+607.4); bf16 2,687.5 MiB. The author's 8-bit isolated core was only ever "a suggested area to look at" (15:03). It is now a candidate, chosen per tensor class by a byte-budget solver (D22) and only where it fits.
* **Derived by.** Bits per weight (`Q4_K` 4.5, `Q6_K` 6.5625, `Q8_0` 8.5) applied to each class's parameter count. Class raises to `Q6_K`: shared experts +25.7, attention +65.5, DeltaNet out +61.9, DeltaNet in +127.8, alpha/beta +1.0 MiB (sum +282.0).
* **Label.** DERIVED. **Depends on it:** the VRAM headroom (1.6). **Master:** 4.4, 9.4, Appendix E.

### 1.6 How much room the core really has next to the engine's bank
* **Finding.** The VRAM ceiling is 10.7 GiB (author ruling 2026-10-23; it was 10.9). With the engine's 7.5 GiB expert bank, its measured whole-card peaks are 10.40–10.44 GiB at the default prefill micro-batch and 10.67–10.69 GiB at `-ub 2048`. With a 0.1 GiB safety margin, the room for any extra core bytes is **0.16 GiB** (default) or **none** (`-ub 2048`, −0.09 GiB); at 10.9 it was 0.36 and about 0.11 GiB. A bank of 8.0 GiB does not fit at all (peaks 11.0–11.19 GiB). **This holds only if the ceiling is read the way the engine's peaks are measured (the whole card, Windows' share included).** Read as a limit on the process's own allocations (how the probe P-H8 and the governor's plan sum see it), the same peaks leave about 1.2 / 0.9 GiB (see 1.18, master 16.3.1 and decision 7).
* **Derived by.** Ceiling − margin − measured peak. This corrected an earlier statement in the work that quoted only the larger figure. For the process-only reading the ≈ 1.0 GiB that Windows already holds before the process starts (0.89 / 1.01 GiB, NVML, WSL2 / Windows-native runs) is subtracted from the peaks first.
* **Label.** DERIVED from the engine's MEASURED peaks. The engine's own allocator may differ: P-E4 re-measures it. **Depends on it:** which core types are allowed, so the export. **Master:** 2.7, 4.3, 9.4, 16.3.1.

### 1.7 The expert stream costs a fixed amount per step, so batch size hides it
* **Finding.** The frozen experts are 19,503,513,600 B (18.164 GiB). They are streamed twice per optimizer step (forward, then backward). At the measured 26.1 GB/s that is **1.49 s per step regardless of how many tokens the step holds** (P-H5 later measured 26.3 GB/s pinned, about 1 % faster, on both platforms). The transfer is hidden behind compute once a step holds at least about 1,495 / 2,242 / 2,989 tokens when compute runs at 1,000 / 1,500 / 2,000 tokens/s.
* **Derived by.** 2 × bytes ÷ bandwidth, then bytes-per-step ÷ compute rate. The bandwidth is the engine project's measured zero-copy floor.
* **Label.** DERIVED from a MEASURED bandwidth. **Depends on it:** the layer-major scheduler with K micro-batches per layer sweep (D5). **Master:** 5, 13, Appendix E.

### 1.8 Work per token and peak speed
* **Finding.** Training costs about 20.39 GFLOP per token at 8,192 tokens (forward 6.74). Frozen weights get no weight-gradient, so backward costs about one forward, not two. Peak rates for this card: bf16 58.3 TFLOPS stock (DERIVED) and, at the author's overclock, 65.2 sustained (MEASURED by P-H3; the derived figure was 65.4), int8 233.2 TOPS stock (DERIVED) and 259.5 sustained (MEASURED; derived 261.4).
* **Derived by.** Per-operation counts in master 6.10; peaks scaled from the published RTX 4090 rows to 46 SMs and this card's clocks.
* **Label.** DERIVED. **The achieved fraction of peak is unknown** (P-H3 has now measured the real peaks; P-K1 to P-K3 measure the efficiency). Speed targets of 600 / 1,200 / 2,000 tokens/s are DECISIONS to be re-based after those probes. **Master:** 6.10, 13.

### 1.9 The 10 full-attention layers do not route differently
* **Finding.** Using the engine's routing trace (84,503 fit and 36,657 held-out decode steps × 40 layers × 8 experts): overall bank hit rate 0.8273, any-miss share 0.639. For the 10 full-attention layers hit rate is 0.8299 against 0.8265 for the other 30; the share of tokens with zero misses is 0.3636 against 0.3602. Holding all 10 layers fully resident would cost 4.605 GiB (4,715 MiB = 4 × 432 + 6 × 498) of the 7.5 GiB bank and would **lower** the overall hit rate to 0.7114.
* **Derived by.** A script over the engine's trace (copy kept with the backups). The coverage curve for those layers at set size k: 64 → 0.118, 128 → 0.519, 192 → 0.831, 224 → 0.941, 256 → 1.0; sizes needed for 90 / 95 / 99 % coverage are 210 / 227 / 247 experts.
* **Label.** MEASURED (the trace) and DERIVED (the analysis). **Consequence.** The premise "full-attention layers need priority residency" is not supported by the engine's own data. It is kept only as an optional, gated experiment (L-Z) with a kill criterion. **Master:** 16.5 B1.1, 16.6 (P-E1).

### 1.10 Training the router for locality is a gated hypothesis
* **Finding.** Mode L trains the router (21.5 M fp32 parameters, experts frozen) with a term that rewards picking experts that sit in the engine's VRAM bank. Cost: 82.0 MiB per copy, +0.32 GiB with gradient, Adam and a frozen reference, and 0.206 % of the FLOPs. Rule of thumb from the engine's bank sweep: about +1 token/s per +0.01 hit rate.
* **Derived by.** Parameter and FLOP arithmetic (41 × 2048 × 256; 41.9 MFLOP per token weight-gradient); the exchange rate from the engine's measured hit-rate-versus-bank-size curve.
* **Label.** Costs DERIVED; benefit **UNVERIFIED**. The literature is mixed: one paper's stacking result and another paper's router-only fine-tune support it, and the first paper's headline result contradicts it at small scale. It ships only if four pre-declared gates pass: quality within +1 % cross-entropy, hit rate up by at least 0.03, breadth across domains, and a measured engine speed gain. Post-hoc clamping of the router at inference is **not used** (the author's ruling; the engine blueprint's description of the paper it cites for this is one of the corrections proposed in master 16.7). **Master:** 7.12, 16.5 B1.

### 1.11 A possible engine-side lever: optimise zero-miss share, not byte-weighted hit
* **Finding.** The engine's bank is filled by water-filling that maximises byte-weighted hit rate. If a layer's cost jumps when it has *any* miss (a large fixed cost c0 against a small per-miss cost c1), then maximising the share of layer-tokens with *zero* misses could beat maximising hits.
* **Derived by.** Reasoning from the any-miss share (0.639) against the hit rate (0.8273) in 1.9.
* **Label.** **UNVERIFIED.** P-E2 (cost of a layer as a function of its miss count) decides it. **First item for the engine cross-check after the pause.** **Master:** 16.5 B1.7.

### 1.12 Where bytes must sit for the engine's fast reads
* **Finding.** A `Q6_K` block is 210 B (128 + 64 + 16 + 2), a row is 2 blocks = 420 B, which is not a multiple of 16. Padding rows to 432 B costs +2.86 % per row, +120 MiB over the engine's 20 `Q6_K` layers (0.65 % of the expert bytes). A GGUF file cannot hold padded rows, so row padding belongs to the engine's own packed layout, not to the trainer's GGUF. The trainer *can* give every expert tensor a 2 MiB-aligned start at no cost in bytes, by placing the expert tensors last and filling the header with a pad entry (all 123 expert tensors are exact multiples of 2 MiB).
* **Derived by.** Block layout read from ggml source; size arithmetic; the 123 = 41 × 3 tensors read from 1.1.
* **Label.** READ-SRC and DERIVED; the benefit of 16-byte coalesced loads (zero-copy reads 9.59 → toward 26.2 GB/s) is **UNVERIFIED** until P-E10; the benefit of 2 MiB pages until P-E11. **Master:** 16.5 B4, Appendix E.

### 1.13 Accuracy gates: what "KL ≤ 0.017353" can and cannot mean
* **Finding.** The engine's figures 0.017353 (batch 16) and 0.018286 (batch 1024) are the measured noise floor of the **current served file** on the engine's own 16 × 1,024-token set. They are not a universal constant. The trainer therefore uses three gates: **K1** trainer against llama.cpp on the *same* GGUF; **K2** exported GGUF against the trainer's reference; **K3** engine against llama.cpp on the exported GGUF. Before relying on the engine's numbers: reproduce them (P-Q4a), then re-measure the float spread on the new file (P-Q4b).
* **Derived by.** Reading the engine's KL gate ledger and noting that a retrained file has different weights, so a new floor.
* **Label.** MEASURED (the engine's numbers, for the old file); the new file's floor is **UNVERIFIED** until P-Q4a and P-Q4b. **Master:** 16.5 B3, 10.

### 1.14 The "bimodal" worry is real in kind, unproven in size
* **Finding.** The outside proposal assumed fp16/bf16 accumulators would diverge between the CPU and GPU expert branches. The K-quant kernels use **int32 accumulators with fp32 scales**, so that premise is not supported. What does differ is the activation quantization: the CPU path uses 256-element blocks (`Q8_K`), the GPU path 32-element blocks (`q8_1`). Outlier activations could make the two branches disagree more. Router near-ties (8th against 9th logit) are a second source of branch flips.
* **Derived by.** Reading both paths in llama.cpp source (commit `2ca15f540`). Adopt Mode B (bimodal-faithful forward) or Mode N (outlier regulariser, ρ = max|x| / rms(x) per 256-block) only if P-E12 shows correlation r ≥ 0.3 and the top 5 % of ρ carry at least half of the excess KL.
* **Label.** READ-SRC for the arithmetic; effect size **UNVERIFIED** (P-E12, P-E3, P-E13). **Master:** 16.5 B5.

### 1.15 Where the engine stands against its ceiling
* **Finding.** The modelled ceiling is 10.87 ms (92.0 tok/s) to 11.34 ms (88.2 tok/s) per token. Measured: WSL 37.9 tok/s short and 42.1 at 12k context (gap 12.4–15.5 ms); Windows-native 47.3 / 46.8 short and 53.7 / 54.1 at 12k (gap 7.2–10.5 ms). Windows is 5.1–5.2 ms per token faster at decode. A "57.0 tok/s" Windows figure in an outside report is not supported by any recorded run (the author thinks it was mistyped). The best recorded WSL cell is a different configuration (7.5 GiB set, `-ub 2048`: 53.5 short, 53.0 at 4k, 49.0 at 12k, gap 7.4–9.5 ms, measured by a different client), so the 5.1–5.2 ms Windows–WSL difference holds only for the reference cell; the OS effect at equal settings is unmeasured.
* **Derived by.** 1000 ÷ tok/s against the ceiling, using the six logged runs; script kept with the backups.
* **Label.** DERIVED from MEASURED runs. **Master:** 16.4.1.

### 1.16 What happened to the outside research report
* **Finding.** Of 18 claims (G1–G18) from an AI-generated engine research plan: some adopted as hypotheses or tests, some rejected. G16–G18 were added after a second full read (WSL speed range supported only for the best cell; its VMBus explanation covers at most a fifth to a quarter of the measured gap; Windows scheduler fixes are Windows-only hypotheses). Checked: the Windows 4 KB page statement is true and its quote verified (G2); "4 KB pages cause the 2× prefill anomaly" is **contradicted** by the engine's own data, because the fast case is ordinary anonymous memory and transparent huge pages are already set to always (G4); an `nvidia-smi -lgc 2475` suggestion would underclock the author's overclock and was rejected.
* **Label.** Each claim carries its own label in the master. **Master:** 16.4.

### 1.17 Things deliberately not built, with the reason
* **Expert consolidation** (merging experts so a layer fits the bank): 128 experts × 3 × 2048 × 512 per layer over 10 layers is 4.03 B parameters, so fp32 masters need 16.1 GB and Adam moments 32.2 GB. Not feasible here. (DERIVED; master 16.5 B1.6.)
* **Router clamping at inference:** excluded by the author's ruling.
* **PyTorch / PEFT / Unsloth code inside the product:** excluded by the ground-up rule; no AGPL or LGPL code is copied.
* **A claim of novelty:** none is made in the repositories.

### 1.18 What the machine measured (Wave 0 hardware set, 2026-10-23/24)
* **Finding 1 — the ceiling.** The card has 11.99 GiB, but once a CUDA context exists CUDA reports only **10.83 GiB free on WSL2 and 10.85 on Windows-native**, identically on every run and in 21 engine logs (a fixed driver budget; inference). The first P-H8 rule (free ≥ ceiling + 0.3 = 11.2) could never pass, so the author set the ceiling to 10.7 and the rule became "the smaller of CUDA free and NVML free after the context ≥ 10.7". The "11.2" in the author's memory is most likely the engine's own gate, "VRAM at most 11.2 GiB" (P1 and G3 in its benchmark files), which is a **whole-card** peak; the 8.0 GiB bank peaked at 11.19 GiB and ran. So three different quantities have carried the name "the ceiling" (master 16.3 X15), and which one the 10.7 means is **decision 7**.
* **Finding 2 — copy/compute overlap needs a flush.** On WSL2 and on Windows-native, work queued on different streams is not sent to the GPU until something flushes the queue; copies and kernels then run one after another (overlap efficiency 0.47–0.52), and a `cudaStreamQuery` after each enqueue brings it to 0.92 (the copy pair alone 0.84). The flush costs 8 µs of host time per call on Windows-native and 77 µs on WSL2, hidden while the GPU has queued work. The streamer therefore flushes once per multi-millisecond batch (D27). Whether the **engine** suffers the same loss is unmeasured; the pathway is to add the flush to its enqueues and compare token rates (engine side, held).
* **Finding 3 — the platforms agree on the GPU and differ on the host.** Tensor rates (bf16 65.2, int8 259.5 T(FL)OPS, 99.8 % of burst, flat clock for 30 minutes), VRAM bandwidth (523–534 GB/s read) and PCIe (26.3 GB/s each way) are the same on both; flush cost, pinned-allocation time (0.8–1.0 s vs 0.07 s per GiB), small-copy latency and the memory clock under load differ. 32 GiB of pinned memory was held on both (the limit is not found), which retires risk R7.
* **Finding 4 — the throttle-reason field is useless here.** The driver reports "gpu_idle" on every sample under full load on both platforms, so thermal judgement uses the SM clock and temperature.
* **Finding 5 — a long GPU job did not disturb the display.** Five stages (kernels of 2, 5 and 30 s, a 10 s queued set, a 10 s graph; about 60 W) completed on both paths with no display-driver events and nothing seen on screen. Heavier kernels and other display setups are unmeasured.
* **Derived by.** One standalone program (`pelias_probe`) run in full on each platform; each probe writes one JSON; the numbers above are read from those files, not from memory. The raw files are in the author's backup folder, not in the repository.
* **Label.** MEASURED, except the "fixed driver budget" and "most likely the engine's gate" (inference). **Not yet measured, with pathways:** P-M4 (needs the model path), the stock-clock repeat, the Windows jitter number (the probe asks Windows for no finer timer, so its 6 ms is probably the 15.6 ms default tick), pinned memory above 32 GiB, the memory-clock difference, ReBAR on/off, heavier kernels for the display test, the Balanced power plan, the engine's own overlap. **Master:** 11.12, 5.4, 15.1, 16.3 X15–X16, D26, D27.

### 1.19 Universal use: the author's design points (2026-10-24; not built)
* **Finding.** The author wants the trainer to adapt to the machine and to the MoE the user chooses. Six points were recorded (master 3.3): (U1) a quick probe at every start writes a machine profile, with the slow probes run once per machine and cached; (U2) the ceiling comes from the probe results, not a typed constant (this is the ruling on decision 7); (U3) a plan card after the probes gives the user a real choice between two to four feasible plans, each with measured fit, labelled speed, quality risk and export effect; (U4) a different model is identified by its file's SHA-256 plus the shapes in its metadata, and the per-model facts are stored under that key; (U5) if a probe fails from a memory spike, a rerun with a flush shows whether flushing helps, and if it does training flushes too; (U6) before the plan card the system scans what holds GPU memory and proposes which apps to close (close all, choose which, or none), then re-measures and recomputes the ceiling.
* **What exists today.** The base file's SHA-256 is recorded as provenance only (checkpoints, export report, layout manifest, engine pack). The run scripts gained a GPU-memory log, a `SETTLE` wait and a `RETRY_ON_FAIL` rerun (master 11.1 rule 8). No real failure has occurred, so the retry has not been used.
* **Derived by.** Reading the author's statements against the master. Nothing here is a measurement.
* **Label.** DECISION (D28, D29). The hash time is UNVERIFIED: it depends on the read rate (P-M4) and the CPU's SHA-256 speed, both unmeasured. The drift of other programs' GPU memory during a run is unmeasured too; the way to measure it (the disturbance test) is in master 11.12. **Master:** 3.3, 11.1, 11.12, 15.1, 16.3.1, 16.7 decision 7, D28, D29.

---

## 2. How the pieces depend on each other

### 2.1 The main chain (read left to right: an arrow means "needs")

```
served GGUF bytes (1.1) ──► core size by type (1.5) ──► byte budget (D22, master 9.4) ──► export of the core
                                     ▲                                  ▲
 engine bank size 7.5 GiB ──► VRAM headroom (1.6) ──────────────────────┘
        ▲
 engine hit-rate-versus-bank curve (MEASURED) ──► value of locality (1.10) ──► Mode L gates
        ▲                                                    ▲
 routing trace (P-E1, 1.9) ─────────────────────────────────┘── decides L-Z and the zero-miss lever (1.11, P-E2)

expert stream bytes (1.7) + bandwidth (MEASURED) ──► tokens per step needed ──► scheduler (D5)
FLOP per token (1.8) + peak rates (P-H3) + efficiency (P-K1..K3) ──► tok/s ──► speed targets

KL floor of the served file (1.13) ──► K1/K2/K3 gates ──► whether the core's type and Mode Q are accepted
 Q6_K row geometry (1.12) ──► alignment plan ──► engine repack (the engine owns the packed layout)
```

### 2.2 If this input changes, these results must be redone

| If this changes… | …then redo | Because |
|---|---|---|
| The engine's bank size (7.5 GiB) | 1.6 headroom → 1.5 core types → export; 1.10 value of locality | The core's room and the hit rate both move |
| The prefill micro-batch (`-ub`) | 1.6 (0.16 GiB or none on the whole-card reading) | The engine's peak VRAM changes |
| The llama.cpp commit (`2ca15f540`) | 1.2, 1.12, 1.14, the quantizer port (P-E6) | Graph, block layouts and kernels are read from that commit |
| The served GGUF (a new trained file) | 1.1, 1.13 (new KL floor), 1.9 (routing may shift) | Weights and routing change |
| Measured expert bandwidth (26.1 GB/s engine; 26.3 GB/s pinned, P-H5) | 1.7 tokens-per-step thresholds | The stream time is bytes ÷ bandwidth |
| Clock settings (overclock on or off) | 1.8 peak rates; P-H3 | Rates scale with clocks |
| The VRAM ceiling (10.7 GiB, author-stated) and what it is measured against | 1.6 and the memory governor (master 15) | Everything is budgeted against it |
| The GPU driver (617.42) or the submission pattern | 1.18 (flush rule), P-H12, P-H9 | Copy/compute overlap depends on when the driver sends queued work |
| The engine adds `Q8_0` kernels (P-E5) | 1.5 (the 8-bit candidate becomes usable) | The engine must be able to run what the trainer exports |

### 2.3 Which section of the master holds which derivation

| Topic | Master section | Probes that replace an estimate |
|---|---|---|
| Model facts and byte map | 2.2–2.5, Appendix A | none (READ-SRC) |
| Baseline being replaced (v0.2.1) | 2.6 | P-T9 (new trainer on v0.2's recipe) |
| Precision map and export | 4, 9.4 | P-E6, P-E7, P-E8 |
| Kernels | 6 | P-H3 (done, 11.12), P-K1–K3 |
| Router locality (Mode L) | 7.12, 16.5 B1 | P-E1 (done), P-E2, P-E3, P-L1–L10 |
| Verification ladder | 10 | P-Q4a, P-Q4b |
| Engine fit | 16 | P-E4, P-E5, P-E9–P-E14 |
| Risks and kill criteria | 14 | per risk |
| Measured hardware facts | 11.12, 2.1 | the hardware set has run except P-M4; leftovers listed in 11.12 |
| Universal use (start-up probe, plan card, model key) | 3.3, D28, D29 | none yet; the parts are P-H3, P-H5, P-H8, P-H12, P-M3, P-M4 |
| All arithmetic | Appendix E | none (scripted) |

---

## 3. What is needed

### 3.1 Decisions needed from the author (master 16.7)

| # | Decision | Why it matters | What it changes |
|---|---|---|---|
| 1 | Is the 7.5 GiB bank a ruling or a derived value? | The core's room (1.6) is set by it; 8.0 GiB does not fit | Core types, export size |
| 2 | Which prefill micro-batch ships? | 0.16 GiB of room at the default, none at `-ub 2048` (whole-card reading; it was 0.36 / about 0.11 at 10.9) | Which core classes can be raised above their served type |
| 3 | Will the engine add `Q8_0` kernels? | The 8-bit core is only usable if the engine can run it | Whether the 8-bit candidate stays on the list |
| 4 | Promote Windows-native from "not in scope"? | It is 5.1–5.2 ms per token faster at decode (1.15) | Which platform the trainer's exports are tuned for |
| 5 | Is Mode L / L-Z worth building after the Wave 0 probes? | Benefit is unverified and the evidence is mixed (1.9, 1.10) | Whether any router training is built |
| 6 | Is there a Windows run newer than the repo's? | The gap table uses the recorded runs only | 1.15 |
| 7 | What is the 10.7 GiB ceiling a ceiling of? **Ruled 2026-10-24 by the author: it comes from the probe results at start** (process-only reading; the follow-on that keeps 10.7 for this machine until the margin is measured and keeps the whole-card figures as the stricter planning figures is a proposal, not yet agreed) | Whole card: 0.16 GiB / none for the core. Process only: about 1.2 / 0.9 GiB and the 8-bit core would fit (1.18, master 16.3.1) | Core types, export size, and whether the engine's bank is re-examined |

### 3.2 Measurements needed (each has a probe in master Chapter 11; Wave 0 needs no trainer code)

**Speed foundation** — without these the speed story is arithmetic on assumed efficiency:
* P-H3 real tensor rates and clocks; P-H8 free VRAM at start (both done 2026-10-23/24, master 11.12).
* P-K1 efficiency of the bf16 path; P-K2 efficiency of the int8 path; P-K3 whether in-register dequantization is memory-bound or compute-bound.

**Accuracy foundation** — without these a "close to llama.cpp" claim cannot be made:
* P-E6 is our port of the ggml quantizers bit-exact for `Q4_K`, `Q6_K`, `Q8_0`?
* P-E7 do the served weights sit exactly on their grid?
* P-E8 does Mode Q close the merge gap, and at what cost?
* P-Q4a reproduce the engine's KL floor; P-Q4b re-measure the float spread on the new file.
* P-E13 which accumulator types and reduction trees do the engine's kernels and llama.cpp's use?

**Engine fit** — these decide how the two repositories lock together:
* P-E2 cost of a layer against its miss count (decides the zero-miss lever, 1.11).
* P-E3 logit margin between the 8th and 9th expert (is zero-spill feasible for this router at all?).
* P-E4 how much VRAM the engine really leaves, per bank size and context.
* P-E5 which weight types the engine's kernels will support.
* P-E9 alignment of the served file's tensors, rows and blocks.
* P-E10 do 16-byte staged loads lift zero-copy reads from 9.59 toward 26.2 GB/s, and does `Q6_K` need row padding?
* P-E11 do 2 MiB pages change zero-copy read speed?
* P-E12 does the CPU/GPU expert-output difference grow with activation outliers?
* P-E14 does the trainer's hot-set writer reproduce the engine's file byte for byte?

**Baseline equivalence:**
* P-T9 does the new trainer, on v0.2's recipe, learn at least as well as v0.2?

### 3.3 Information needed from outside this repository
* The author's new engine information (pending at the pause).
* Any Windows-native run newer than the one in the engine repo (decision 6).
* The engine project's own cost-per-miss data, or permission to run P-E2 (1.11).
* The original outside research report, if it is to be kept with the record (the master holds a claim-by-claim table of it, 16.4).

### 3.4 Work not yet done
* **No trainer code** exists. The stages T0–T14 (master 12) have not started. The hardware probe program exists and has run (P-M4 is built but has not run); every other probe is still a specification.
* **Edits to the engine blueprint** are listed in master 16.7 and are **proposed, not made**: scope line; VRAM-budget table; CPU-read row; E3 gate (the oracle reads trainer GGUFs); E5 (per-layer capacities, `Q6_K` 210-byte blocks); E11 (huge pages); E12 (hot-set objective); unknowns table; contract section; sources. They wait for the author to lift the pause.
* **Engine side:** the cross-check against the speed ceiling, the engine master notes, the engine plain-English README, and the engine repository update.

### 3.5 Weak points in this design (so they are not missed)
* **The margin under the measured ceiling is unmeasured** (decision 7 is ruled; the disturbance test in master 11.12 sets the margin). Until then the core's room is 0.16 GiB / none on the whole-card figures or about 1.2 / 0.9 GiB on the process-only figures.
* **Efficiency is unmeasured.** Every tokens-per-second figure for the new trainer is conditional on it (1.8).
* **The size of the merge loss on the core is unmeasured** (1.4); the whole precision plan has an escalation ladder (master 4.8) for the case it is worse than expected.
* **The hot-set and locality ideas are hypotheses** (1.9–1.11). Nothing here depends on them to produce a working trainer.
* **The author's guess that 70–80 % of the earlier draft was compliant is not confirmed.** The compliance matrix has 22 rows: 7 already met, 4 needing correction, 6 new specification, 3 resting on premises the engine's data does not support, 2 waiting on probes. There is no natural denominator for a percentage.
* **Label counts in the master** (occurrences, 2026-10-24): MEASURED 36 · READ-SRC 42 · PUBLISHED 33 · DERIVED 83 · ESTIMATE 23 · UNVERIFIED 21 · DECISION 24. Each ESTIMATE and UNVERIFIED claim is paired with a probe.

### 3.6 How to keep this file true
When a probe returns, change its entry's label from ESTIMATE or UNVERIFIED to MEASURED, record the date and the raw file, and update the rows of 2.2 that depend on it. Never replace a derived number with a guess. If a result contradicts the master, correct the master first, then this file.
