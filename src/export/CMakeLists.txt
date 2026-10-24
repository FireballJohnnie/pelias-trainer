# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# export: Adapter GGUF, merged-core GGUF, splice with hash proof.
# pelias-deps: io model
pelias_module(export
    CC export_adapter.cc export_core.cc splice.cc
    DEPS io model)
