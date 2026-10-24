# Third-party code

Nothing is vendored yet. Pins to use when it is needed (master 5.1):

* llama.cpp, commit `2ca15f540` (MIT): libllama and `common`, for the tokenizer and the chat template.
* nlohmann/json (MIT): a small JSON parser, already vendored by llama.cpp.

Anything copied or adapted must say so in its first line (`// provenance:`) and be listed in `CITATIONS.md`.
