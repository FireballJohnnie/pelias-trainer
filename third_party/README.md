# Third-party code

Nothing is vendored, and nothing from llama.cpp, ggml or a JSON library is linked or copied into the product (master D37, "ground up").

Oracle pin, for tests and comparisons only (never linked):

* llama.cpp, commit `2ca15f540` (MIT): its tokenizer, chat-template renderer, quantizers, `convert_hf_to_gguf.py` and `llama-quantize` are the references that our own code is compared with (master 10.6, P-D1, P-D2, P-E6, P-X0, P-X6).

Where our bytes must equal ggml's (the quantizer, the block layouts), the code is original and its first line (`// provenance:`) says that it reproduces ggml's arithmetic or layout. Anything copied or adapted must say so in its first line and be listed in `CITATIONS.md`.
