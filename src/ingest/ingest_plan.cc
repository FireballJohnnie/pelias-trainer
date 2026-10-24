// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// ingest/ingest_plan.cc: see ingest_plan.h.
#include "ingest/ingest_plan.h"

#include <filesystem>
#include <system_error>

namespace pelias::ingest {

namespace fs = std::filesystem;

namespace {

PlanCheck bad(const char* why) { return PlanCheck{false, why}; }

bool has_safetensors(const fs::path& dir) {
    std::error_code ec;
    for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == ".safetensors") return true;
    }
    return false;
}

// True when `inner` is `outer` or lies inside it.
bool is_inside(const fs::path& inner, const fs::path& outer) {
    auto i = inner.begin();
    for (auto o = outer.begin(); o != outer.end(); ++o, ++i) {
        if (i == inner.end() || *i != *o) return false;
    }
    return true;
}

}  // namespace

PlanCheck validate_plan(const IngestPlan& plan) {
    if (plan.source_dir.empty()) return bad("the model folder is not set");
    if (plan.output_dir.empty()) return bad("the output folder is not set");

    std::error_code ec;
    fs::path src = fs::weakly_canonical(plan.source_dir, ec);
    if (ec || !fs::is_directory(src, ec)) return bad("the model folder does not exist");
    if (!has_safetensors(src)) return bad("the model folder has no safetensors files");

    fs::path out = fs::weakly_canonical(plan.output_dir, ec);
    if (ec) return bad("the output folder cannot be resolved");
    if (is_inside(out, src)) return bad("the output folder must not be inside the model folder");

    config::Check k = config::validate(plan.cfg);
    if (!k.ok) return PlanCheck{false, k.reason};
    return PlanCheck{};
}

RunResult run_stage_a(const IngestPlan& plan) {
    if (!validate_plan(plan).ok) return RunResult::kInvalidPlan;
    return RunResult::kNotImplemented;
}

}  // namespace pelias::ingest
