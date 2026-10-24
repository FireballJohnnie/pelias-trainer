# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# io: Memory-maps the GGUF, typed block views, splice writer, SHA-256, safetensors interchange.
# pelias-deps: hal
pelias_module(io
    CC gguf_reader.cc gguf_writer.cc sha256.cc safetensors.cc
    DEPS hal)
