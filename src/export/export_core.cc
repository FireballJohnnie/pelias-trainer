// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// export/export_core.cc: see export_core.h. Only decide_export is written.
#include "export/export_core.h"

namespace pelias::exporter {

ExportDecision decide_export(const config::ExportPolicy& policy) {
    ExportDecision d;
    d.delta_max_bytes = policy.delta_max_bytes;
    d.write_adapter = policy.write_adapter;
    d.adapter_is_reference = policy.adapter_is_reference();
    switch (policy.mode) {
        case config::CoreMode::kModeQ:
            d.write_merged_core = true;
            d.core_is_copy_of_last_w_eff = true;
            d.quantize_after_training = false;
            break;
        case config::CoreMode::kMergeRaised:
            d.write_merged_core = true;
            d.core_is_copy_of_last_w_eff = false;
            d.quantize_after_training = true;
            break;
        case config::CoreMode::kAdapterOnly:
            d.write_merged_core = false;
            d.core_is_copy_of_last_w_eff = false;
            d.quantize_after_training = false;
            break;
    }
    return d;
}

}  // namespace pelias::exporter
