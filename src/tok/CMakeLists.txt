# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# tok: Tokenizes and renders exactly like the server (wraps libllama / common).
# pelias-deps: 
pelias_module(tok
    CC tokenizer.cc chat_template.cc)
