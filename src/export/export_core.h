// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// export/export_core.h: merged-core GGUF (master 5.2, 9.3, 9.4).
// Status: only the decision that follows from the configuration is written (decide_export); the writer is NOT IMPLEMENTED.
#pragma once

#include <cstdint>

#include "config/trainer_config.h"

namespace pelias::exporter {

// What one export run does, as a function of the policy (master 9.1, 9.3 step 2, 9.4).
struct ExportDecision {
    bool write_merged_core = true;
    // Mode Q (the default): the core bytes are a copy of the last W_eff; nothing is quantized at export time.
    bool core_is_copy_of_last_w_eff = true;
    // Override path: dequantize the base, add the delta, quantize to the class's raised type.
    bool quantize_after_training = false;
    bool write_adapter = true;
    bool adapter_is_reference = false;
    // The frozen experts are always byte-spliced from the base and hash-proven (D2); no policy switches this off.
    bool splice_experts_with_proof = true;
    // Extra bytes the core may add over its served footprint: 0 by default.
    std::uint64_t delta_max_bytes = 0;
};

// The policy must have passed config::validate; an invalid policy gives an unspecified (but safe) decision.
ExportDecision decide_export(const config::ExportPolicy& policy);

}  // namespace pelias::exporter
