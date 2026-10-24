# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# ingest: Stage A, the original bf16 safetensors to pelias_base_0step.gguf (plan and checks only so far).
# pelias-deps: io config
pelias_module(ingest
    CC ingest_plan.cc
    DEPS io config)
