# Providers

Built-in embedding providers are `local` and `sentence_transformers`. Built-in vector
stores are `sqlite`, `memory`, and `qdrant`. LLM providers are `extractive`,
`openai_compatible`, `openai`, and `ollama`. Cache providers are `memory` and `redis`.

External packages can register embedding, vector-store, or LLM classes through the
`ragzen.plugins` entry-point group. Plugin classes may expose `from_config(config)` or
accept a `config=` constructor argument.
