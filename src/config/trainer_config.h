// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// config/trainer_config.h: the one configuration struct and its defaults (master 4.10, 9.4, 5.3; D31, D32, D34).
//
// The defaults are the author's ruling of 2026-10-24: Mode Q at the served types (+0 extra bytes), the byte budget
// delta_max_bytes = 0, the D22 byte-budget solver and every type raise off, the planner automatic (the start-up probe
// decides the micro-batch and the context), and the Advanced/Developer switch off (the Expert mode of master U11).
// Everything that differs from the defaults needs `advanced = true`.
//
// No hardware number is typed here. The 10.7 GiB ceiling, the window sizes and the slot counts come from the probes;
// the governor in hal stays the hard stop whatever this struct says.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace pelias::config {

// How the core (shared experts, DeltaNet projections, attention projections) is trained and exported.
enum class CoreMode : std::uint8_t {
    kModeQ,        // default (D31, master 4.10): LoRA trained through the exact quantizer; export the last W_eff
    kMergeRaised,  // override (D32): LoRA trained as usual, merged after training at the raised types of D22
    kAdapterOnly,  // release the adapter alone (master E-A); no merged core
};

// The five classes of master 9.4. A class is raised as a whole.
enum class CoreClass : std::uint8_t {
    kSharedExperts,
    kDeltaNetIn,
    kDeltaNetAlphaBeta,
    kDeltaNetOut,
    kAttention,
};
inline constexpr std::size_t kCoreClassCount = 5;

// kServed keeps the type the Stage A type map gives the class (the stock Q4_K_M mix).
enum class CoreType : std::uint8_t {
    kServed = 0,
    kQ6K,
    kQ8_0,
};

struct ExportPolicy {
    CoreMode mode = CoreMode::kModeQ;
    // Bytes the core may add over its served footprint. 0 by default: the export never depends on spare engine VRAM.
    std::uint64_t delta_max_bytes = 0;
    // D22 override: let the byte-budget solver choose the raised classes inside delta_max_bytes.
    bool use_byte_budget_solver = false;
    // D22 override: pin a class to a raised type by hand. All kServed by default.
    std::array<CoreType, kCoreClassCount> pinned_type{};
    // The adapter GGUF is always written unless this is switched off (Advanced/Developer).
    bool write_adapter = true;

    bool any_pinned() const;
    // True when the override of D22 is in use in any form.
    bool raises_types() const { return use_byte_budget_solver || any_pinned(); }
    // Under Mode Q training never saw W_base + dW, so the adapter is informational and the merged file is the reference.
    bool adapter_is_reference() const { return mode != CoreMode::kModeQ; }
};

// Manual values for the planner (D34). Empty means "the quick start-up probe decides".
struct PlannerConfig {
    std::optional<std::uint32_t> window_tokens;
    std::optional<std::uint32_t> windows_per_step;
    std::optional<std::uint32_t> context_tokens;
    std::optional<std::uint32_t> expert_slots;
    std::optional<std::uint64_t> vram_ceiling_bytes;  // the governor (master 15.1) still enforces its own ceiling

    bool any_override() const;
    bool automatic() const { return !any_override(); }
};

struct TrainerConfig {
    bool advanced = false;  // Advanced/Developer (the Expert mode of master U11)
    ExportPolicy export_policy;
    PlannerConfig planner;
};

struct Check {
    bool ok = true;
    std::string reason;  // plain words; never a path
};

// The defaults of the ruling of 2026-10-24.
TrainerConfig default_config();
bool is_default(const TrainerConfig& c);
// Rules that keep the overrides honest; see the test in tests/unit/defaults_tests.cc.
Check validate(const TrainerConfig& c);
// A few plain lines for the plan card and for --show-defaults.
std::string summary(const TrainerConfig& c);

}  // namespace pelias::config
