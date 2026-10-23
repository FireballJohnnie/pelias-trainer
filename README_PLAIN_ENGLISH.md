# Pelias Trainer, in plain English

*This is the short, non-technical version of [`README.md`](README.md). If a word here is unfamiliar, the full README and the design document define it. Nothing here is a new claim: every number comes from the results ledger ([`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)) or from the design document, and each one is labelled there as measured, derived or estimated.*

*Draft. This repository is private. Nothing is released.*

---

## What is this?

Large language models can be taught new behaviour after they are built. That is called **fine-tuning**. Fine-tuning a big model normally needs a data-centre GPU with a lot of memory.

This project fine-tunes one specific large model, **Qwen3.6-35B-A3B**, on **one ordinary gaming graphics card with 12 GB of memory** (an RTX 4070). The model is far too big to fit on that card, so the trainer has to be clever about what lives where.

## The one idea that makes it possible

The model is a **Mixture-of-Experts** model. Think of it as a building with 256 specialist rooms ("experts") on every floor. For each word it reads, the model sends the work to only 8 of those 256 rooms. So most rooms are idle most of the time.

The trainer uses that:

* The small part that is **always used** (the "spine") stays on the graphics card, and this is the part that is actually **trained**.
* The huge collection of **experts is frozen** (never changed), stored in compressed 4-bit form in the computer's ordinary memory, and **streamed to the card one layer at a time**, used, and thrown away.

The result is that a model that needs far more than 12 GB can be trained on a 12 GB card, at the cost of patience.

## What exists today

**Version 0.2.1 works, and it made the model the author uses.** It is written in Python. Two real training runs are logged in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). The first one took 202.7 minutes for 616 training steps, at a median of 247.85 tokens processed per second (**measured**). The second run continued from the first. Together they produced the 4-bit and 5-bit model files described in the ledger.

**Version 0.2.1 is unfinished** (the author's word). It has not been fully cleaned up for release, and some tests have not been re-run for this draft.

## What is new: the plan for the next version

A new document, [`docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`](docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md), is a **complete design for rebuilding the trainer from scratch in C++ and CUDA**, with no Python and no PyTorch inside it.

**Important: this is a design only.** No code for the new trainer has been written, and nothing in the design has been run on the GPU yet. Every number in it is labelled, and every claim that could not be checked has a named test (a "probe") that turns it into a measurement. There are 99 probes.

In everyday terms, the design aims to do five things better than version 0.2.1:

1. **Train against exactly what will be served.** The old trainer worked on a slightly different copy of the compressed model than the one the inference engine later runs. The new design reads the very same model file (a GGUF) that the engine serves, so what is trained is what is used.
2. **Leave the frozen part untouched, byte for byte.** The experts are never re-compressed. They are copied into the finished file exactly as they were, which can be proven by comparing checksums.
3. **Waste less time.** The experts are streamed to the card twice per optimiser step (once going forward, once going backward), which is a fixed amount of data however many words the step holds, and a large enough batch hides that transfer behind the computation. The design estimates how fast this could be and says how sure it is. Those estimates are **derived or estimated, not measured**, and the probes exist to replace them with measurements.
4. **Check itself.** Three independent references are used to confirm the maths, and a ladder of ten checks must pass in order before the next stage starts. Nothing continues after a failed check.
5. **Fit the inference engine.** A chapter of the design (Chapter 16) goes through the engine's requirements one by one: which are already met, which needed correcting, and which rest on assumptions the engine's own measurements do not support. It also covers how the trainer's output file lines up with the memory layout the engine expects.

## One optional idea, with a way to kill it

Chapter 7 and Chapter 16 describe an **optional** experiment: nudging the model's routing so that the experts it picks more often are the ones the inference engine keeps in fast memory. It is listed as a hypothesis, not a result. It is only adopted if it passes four tests set in advance, and the design says what outcome would end the experiment.

## What this project does not claim

* It does not claim the new trainer exists, works, or is faster. The design says what to build and how to prove it.
* It does not claim to work on other models. Today's results come from one machine and one model.
* It does not claim to be the first or the only approach of its kind. It describes what was built and measured and lets others judge.
* It is not a finished, released product. The author decides what is published and when.

## What is deliberately kept out of this repository

Model weights, trained adapters, training data, personal identity or system-prompt text, downloaded caches, access tokens and keys. The training data and the model's identity are private by the author's decision.

## Where to read next

| If you want… | Read |
|---|---|
| The technical overview of what exists | [`README.md`](README.md) |
| The measured results of the two training runs and the export | [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) |
| The full plan for the C++/CUDA rebuild, with its tests | [`docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md`](docs/PELIAS_TRAINER_CUDA_MASTER_SCAFFOLD.md) |
| Sources and licences | [`CITATIONS.md`](CITATIONS.md) and section 7 of `README.md` |
| The companion inference engine | [pelias-engine](https://github.com/FireballJohnnie/pelias-engine) |

## Who is behind it

Author and maintainer: John Lipka. Built with AI assistants (Claude and Gemini) under his direction. The hardware measurements were run by the author.
