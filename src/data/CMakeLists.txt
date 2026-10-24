# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# data: Dataset reading, packing, assistant-only masks, audits (counts only).
# pelias-deps: io tok
pelias_module(data
    CC dataset.cc packer.cc masks.cc audit.cc
    DEPS io tok)
