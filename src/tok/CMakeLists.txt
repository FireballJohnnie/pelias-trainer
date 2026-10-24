# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# tok: Tokenizes and renders exactly like the server; own code, no libllama (D37).
# pelias-deps: 
pelias_module(tok
    CC tokenizer.cc chat_template.cc)
