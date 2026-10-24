# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# model: ModelSpec from GGUF metadata, frozen store, trainable core, LoRA, routing tape.
# pelias-deps: io
pelias_module(model
    CC spec.cc frozen_store.cc trainable_core.cc
    CU lora.cu tape.cu
    DEPS io)
