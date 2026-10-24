// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// Unit tests for the defaults of the ruling of 2026-10-24 (master D31 to D34): the configuration struct, the export decision
// that follows from it, and the Stage A plan checks. Host-only; no GPU and no model file.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "config/trainer_config.h"
#include "export/export_core.h"
#include "ingest/ingest_plan.h"

namespace fs = std::filesystem;
using namespace pelias;

static int g_failed = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);                  \
            ++g_failed;                                                          \
        }                                                                        \
    } while (0)

static config::TrainerConfig advanced() {
    config::TrainerConfig c;
    c.advanced = true;
    return c;
}

static void test_defaults() {
    config::TrainerConfig c = config::default_config();
    CHECK(config::is_default(c));
    CHECK(config::validate(c).ok);
    CHECK(c.export_policy.mode == config::CoreMode::kModeQ);
    CHECK(c.export_policy.delta_max_bytes == 0);
    CHECK(!c.export_policy.use_byte_budget_solver);
    CHECK(!c.export_policy.any_pinned());
    CHECK(c.export_policy.write_adapter);
    CHECK(!c.export_policy.adapter_is_reference());  // informational under Mode Q
    CHECK(c.planner.automatic());
    CHECK(!c.advanced);
    const std::string s = config::summary(c);
    CHECK(s.find("Mode Q") != std::string::npos);
    CHECK(s.find("+0 extra bytes") != std::string::npos);
    CHECK(s.find("informational") != std::string::npos);
    CHECK(s.find("type raises: off") != std::string::npos);
    CHECK(s.find("planner: automatic") != std::string::npos);
}

static void test_export_decision_default() {
    exporter::ExportDecision d = exporter::decide_export(config::default_config().export_policy);
    CHECK(d.write_merged_core);
    CHECK(d.core_is_copy_of_last_w_eff);
    CHECK(!d.quantize_after_training);
    CHECK(d.write_adapter);
    CHECK(!d.adapter_is_reference);
    CHECK(d.splice_experts_with_proof);
    CHECK(d.delta_max_bytes == 0);
}

static void test_overrides_need_advanced() {
    config::TrainerConfig c;
    c.export_policy.use_byte_budget_solver = true;
    c.export_policy.delta_max_bytes = 1;
    CHECK(!config::validate(c).ok);  // advanced is off
    c.advanced = true;
    CHECK(config::validate(c).ok);

    config::TrainerConfig p;
    p.planner.window_tokens = 4096;
    CHECK(!config::is_default(p));
    CHECK(!config::validate(p).ok);
    p.advanced = true;
    CHECK(config::validate(p).ok);
    CHECK(!p.planner.automatic());
}

static void test_raises_need_a_budget() {
    config::TrainerConfig c = advanced();
    c.export_policy.pinned_type[static_cast<std::size_t>(config::CoreClass::kAttention)] = config::CoreType::kQ6K;
    CHECK(c.export_policy.any_pinned());
    CHECK(c.export_policy.raises_types());
    CHECK(!config::validate(c).ok);  // no byte budget
    c.export_policy.delta_max_bytes = 1;
    CHECK(config::validate(c).ok);
    CHECK(config::summary(c).find("type raises: on") != std::string::npos);
}

static void test_budget_without_raise_is_refused() {
    config::TrainerConfig c = advanced();
    c.export_policy.delta_max_bytes = 1;  // Mode Q alone adds no bytes
    CHECK(!config::validate(c).ok);
}

static void test_merge_at_served_type_is_refused() {
    config::TrainerConfig c = advanced();
    c.export_policy.mode = config::CoreMode::kMergeRaised;  // E-D: the delta is erased
    CHECK(!config::validate(c).ok);
    c.export_policy.use_byte_budget_solver = true;
    c.export_policy.delta_max_bytes = 1;
    CHECK(config::validate(c).ok);
    exporter::ExportDecision d = exporter::decide_export(c.export_policy);
    CHECK(d.write_merged_core);
    CHECK(!d.core_is_copy_of_last_w_eff);
    CHECK(d.quantize_after_training);
    CHECK(d.adapter_is_reference);  // the adapter is the reference once Mode Q is off
}

static void test_adapter_only() {
    config::TrainerConfig c = advanced();
    c.export_policy.mode = config::CoreMode::kAdapterOnly;
    CHECK(config::validate(c).ok);
    exporter::ExportDecision d = exporter::decide_export(c.export_policy);
    CHECK(!d.write_merged_core);
    CHECK(d.write_adapter);
    CHECK(d.adapter_is_reference);
    c.export_policy.write_adapter = false;
    CHECK(!config::validate(c).ok);
}

static void test_planner_zero_values() {
    config::TrainerConfig c = advanced();
    c.planner.expert_slots = 0;
    CHECK(!config::validate(c).ok);
    c.planner.expert_slots = 2;
    CHECK(config::validate(c).ok);
}

static void test_ingest_plan() {
    const fs::path root = fs::temp_directory_path() / "pelias_defaults_tests_ingest";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "model");
    fs::create_directories(root / "out");

    ingest::IngestPlan plan;
    plan.source_dir = (root / "model").string();
    plan.output_dir = (root / "out").string();
    CHECK(!ingest::validate_plan(plan).ok);  // no safetensors yet
    CHECK(ingest::run_stage_a(plan) == ingest::RunResult::kInvalidPlan);

    std::ofstream((root / "model" / "model-00001-of-00001.safetensors").string()).put('x');
    CHECK(ingest::validate_plan(plan).ok);
    CHECK(ingest::run_stage_a(plan) == ingest::RunResult::kNotImplemented);

    ingest::IngestPlan inside = plan;
    inside.output_dir = (root / "model" / "cache").string();
    CHECK(!ingest::validate_plan(inside).ok);

    ingest::IngestPlan unset;
    CHECK(!ingest::validate_plan(unset).ok);

    ingest::IngestPlan missing = plan;
    missing.source_dir = (root / "does_not_exist").string();
    CHECK(!ingest::validate_plan(missing).ok);

    ingest::IngestPlan raised = plan;  // the override needs Advanced/Developer
    raised.cfg.export_policy.use_byte_budget_solver = true;
    raised.cfg.export_policy.delta_max_bytes = 1;
    CHECK(!ingest::validate_plan(raised).ok);
    raised.cfg.advanced = true;
    CHECK(ingest::validate_plan(raised).ok);

    CHECK(std::string(ingest::kBaseFileName) == "pelias_base_0step.gguf");
    fs::remove_all(root, ec);
}

int main() {
    test_defaults();
    test_export_decision_default();
    test_overrides_need_advanced();
    test_raises_need_a_budget();
    test_budget_without_raise_is_refused();
    test_merge_at_served_type_is_refused();
    test_adapter_only();
    test_planner_zero_values();
    test_ingest_plan();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("defaults_tests: all checks passed\n");
    return 0;
}
