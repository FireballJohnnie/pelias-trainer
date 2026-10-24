// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// config/trainer_config.cc: defaults, validation and summary (see trainer_config.h).
#include "config/trainer_config.h"

namespace pelias::config {

static_assert(static_cast<int>(CoreType::kServed) == 0, "the all-served default relies on value-initialisation");

bool ExportPolicy::any_pinned() const {
    for (CoreType t : pinned_type) {
        if (t != CoreType::kServed) return true;
    }
    return false;
}

bool PlannerConfig::any_override() const {
    return window_tokens || windows_per_step || context_tokens || expert_slots || vram_ceiling_bytes;
}

TrainerConfig default_config() { return TrainerConfig{}; }

bool is_default(const TrainerConfig& c) {
    const ExportPolicy& e = c.export_policy;
    return !c.advanced && e.mode == CoreMode::kModeQ && e.delta_max_bytes == 0 && !e.use_byte_budget_solver &&
           !e.any_pinned() && e.write_adapter && !c.planner.any_override();
}

namespace {
Check bad(const char* why) { return Check{false, why}; }
}  // namespace

Check validate(const TrainerConfig& c) {
    const ExportPolicy& e = c.export_policy;
    const PlannerConfig& p = c.planner;

    if (!is_default(c) && !c.advanced) {
        return bad("a value differs from the defaults but Advanced/Developer is off");
    }

    // Export policy.
    if (e.mode == CoreMode::kAdapterOnly) {
        if (e.raises_types() || e.delta_max_bytes != 0) return bad("adapter-only release cannot raise core types or carry a byte budget");
        if (!e.write_adapter) return bad("adapter-only release needs the adapter to be written");
    } else if (e.raises_types()) {
        if (e.delta_max_bytes == 0) return bad("raising core types needs a byte budget above zero");
    } else {
        if (e.delta_max_bytes != 0) return bad("a byte budget needs the byte-budget solver or a pinned type; Mode Q alone adds no bytes");
        if (e.mode == CoreMode::kMergeRaised) {
            return bad("merging at the served type erases the learned change (master 4.3, E-D); raise a class or use Mode Q");
        }
    }

    // Planner overrides.
    if (p.window_tokens && *p.window_tokens == 0) return bad("window size must be above zero");
    if (p.windows_per_step && *p.windows_per_step == 0) return bad("windows per step must be above zero");
    if (p.context_tokens && *p.context_tokens == 0) return bad("context length must be above zero");
    if (p.expert_slots && *p.expert_slots == 0) return bad("expert slots must be above zero");
    if (p.vram_ceiling_bytes && *p.vram_ceiling_bytes == 0) return bad("VRAM ceiling must be above zero");

    return Check{};
}

std::string summary(const TrainerConfig& c) {
    const ExportPolicy& e = c.export_policy;
    std::string s;
    switch (e.mode) {
        case CoreMode::kModeQ:
            s += "core: Mode Q, trained through the exact quantizer at the served types";
            s += e.raises_types() ? " (some classes raised by the override)\n" : " (+0 extra bytes)\n";
            break;
        case CoreMode::kMergeRaised:
            s += "core: LoRA merged after training at raised types (override)\n";
            break;
        case CoreMode::kAdapterOnly:
            s += "core: adapter only, no merged core (override)\n";
            break;
    }
    s += e.write_adapter ? (e.adapter_is_reference() ? "adapter: written, the reference\n" : "adapter: written, informational\n")
                         : "adapter: not written\n";
    s += e.raises_types() ? "type raises: on (byte-budget override)\n" : "type raises: off\n";
    s += c.planner.automatic() ? "planner: automatic (from the start-up probe)\n" : "planner: manual values set\n";
    s += c.advanced ? "advanced/developer: on\n" : "advanced/developer: off\n";
    return s;
}

}  // namespace pelias::config
