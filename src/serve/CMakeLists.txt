# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# serve: Server command for the finished GGUF. Held until step 5 of the governing order (unify).
# pelias-deps: project
pelias_module(serve
    CC serve_command.cc
    DEPS project)
