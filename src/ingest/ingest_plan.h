// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// ingest/ingest_plan.h: Stage A, the one-time step from the original bf16 safetensors to the 0-step base GGUF (master 4.2, 12.2 T1b; D33).
// Status: the plan and its checks are written; run_stage_a is NOT IMPLEMENTED (it needs the P-E6 quantizer port and the safetensors header).
#pragma once

#include <string>

#include "config/trainer_config.h"

namespace pelias::ingest {

// The cached base file written by Stage A and read by Stage B.
inline constexpr const char* kBaseFileName = "pelias_base_0step.gguf";

struct IngestPlan {
    // Folder holding the bf16 safetensors. The user supplies it at run time; the trainer never downloads a model, and the
    // path is never written to a result file.
    std::string source_dir;
    // Folder that receives (and caches) pelias_base_0step.gguf.
    std::string output_dir;
    // The stock Q4_K_M type map unless the D22 override pins classes (cfg.export_policy, Advanced/Developer only).
    config::TrainerConfig cfg;
    // An existing base file with matching provenance is reused instead of rewritten.
    bool reuse_cached = true;
};

struct PlanCheck {
    bool ok = true;
    std::string reason;  // plain words; never a path
};

PlanCheck validate_plan(const IngestPlan& plan);

enum class RunResult { kOk, kInvalidPlan, kNotImplemented };

RunResult run_stage_a(const IngestPlan& plan);

}  // namespace pelias::ingest
