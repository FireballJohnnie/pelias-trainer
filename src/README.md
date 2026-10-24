# Source tree

Framework only: every file is a stub that names its place and says what it will hold. Nothing here computes anything yet. The exceptions: `config` (the defaults and their checks), `ingest` (the Stage A plan and its checks), `export_core` (the export decision that follows from the configuration) and `ui`, whose files are real: the Dear ImGui drawing layer (spear progress bar, warning window, telemetry). It builds only with `-DPELIAS_UI=ON`.

The product name is `Pelias-Trainer-v<VERSION>-<backend>`, for example `Pelias-Trainer-v0.0.0-cuda`. The backend is `cuda` when the build has CUDA and `host` for the host-only build used to check the framework without a GPU. The version number is not chosen yet (`VERSION` holds a placeholder; the Python trainer is 0.2.1).

## Modules

"In 5.2" says whether the module is in the module tree of master section 5.2. The **new** modules carry the universal-use points U1 and U3 to U11 (master 3.3); they are proposals for where that logic lives and are not yet in the master's tree.

| Module | In 5.2 | May use | Responsibility |
|---|---|---|---|
| `hal` | yes | none | Device selection, stream/event wrappers, arena allocator, pinned pool, VRAM governor, sync audit. |
| `config` | yes | none | The one configuration struct and its defaults: Mode Q at the served types, no extra bytes, type raises off, planner automatic; overrides only in Advanced/Developer (master D31, D32, D34). |
| `tok` | yes | none | Tokenizes and renders exactly like the server (wraps libllama / common). |
| `io` | yes | hal | Memory-maps the GGUF, typed block views, splice writer, SHA-256, safetensors interchange. |
| `ingest` | yes | io, config | Stage A: the original bf16 safetensors to the cached `pelias_base_0step.gguf` (master D33). So far only the plan and its checks are written; the conversion and the quantizer are not. |
| `plan` | yes | hal | Reads the probe constants, solves the plan, prints it before any allocation. |
| `data` | yes | io, tok | Dataset reading, packing, assistant-only masks, audits (counts only). |
| `model` | yes | io | ModelSpec from GGUF metadata, frozen store, trainable core, LoRA, routing tape. |
| `stream` | yes | hal | Expert streamer (pinned store to device slots) and boundary store. |
| `kern` | yes | hal | All device code. Depends only on hal and plain structs of pointers and shapes; never includes model. |
| `sched` | yes | kern, model, stream | The only place that composes kernels into layers; the layer-major step, enqueue-only. |
| `train` | yes | sched, model, data | Optimizer, LR schedule, checkpoints, metrics, the trainer loop. |
| `export` | yes | io, model, config | Adapter GGUF, merged-core GGUF, splice with hash proof. |
| `verify` | yes | hal, io | Gate runner, canaries, KL tools, dump comparison. |
| `project` | **new** | io | The one project file, run folders under the user's output folder, the model key. |
| `startup` | **new** | hal, plan, io, project | Start-up quick pass, machine profile, scan of what holds GPU memory. |
| `intake` | **new** | data, tok | Persona, parameter and client sheets; import of premade lists; intake checks (counts only). |
| `help` | **new** | none | Reads resources/help/*.txt (short, plain, detail, see) for the interface; no interface code here. |
| `warn` | **new** | data, verify, intake | Cautions about fit, loops and a model that does not work; the client matrix. |
| `flow` | **new** | plan, startup, project, intake, warn, verify, help | Propose, confirm, apply to a copy, re-test, final summary, plan card, Expert mode. Logic only; no interface. |
| `serve` | **new** | project | Server command for the finished GGUF. Held until step 5 of the governing order (unify). |
| `ui` | **new** | none | Dear ImGui drawing layer: lithic style, spear progress bar, warning modal, telemetry. So far only the header `pelias_lithic_ui.hpp` is written; the implementation, the Dear ImGui build options and the interface tests are not. Header-only INTERFACE library for now. |
| `cli` | yes | train, export, verify, flow, serve | Command-line entry point. |

The allowed dependencies are kept in `tools/check_module_deps.py` and in each module's `CMakeLists.txt`; the check fails if they disagree, if a file includes a module it may not use, or if the graph has a cycle. `kern` uses only `hal` and plain structs of pointers and shapes.

## Rules for every source file (master 5.1)

* Namespace: `pelias::<module>`. The one exception is `export`, which is a C++20 keyword and cannot name a namespace: its namespace is `pelias::exporter`.

* First line: `// provenance: from the spec: <name>`, `adapted from <X>, licence <Y>` or `original`.
* Second line: `// Copyright (c) 2026 John Lipka and the Pelias contributors`.
* No number is typed into code from memory: rates and limits come from the probe constants or the start-up probes.

## Build the framework (no GPU needed)

```
cmake --preset host-only
cmake --build --preset host-only
ctest --preset host-only
```

Other presets: `wsl-release`, `wsl-debug` (CUDA on, Linux) and `windows-release` (CUDA on; run it from the Visual Studio x64 Native Tools prompt). The CUDA presets have not been run: this framework was built host-only.
