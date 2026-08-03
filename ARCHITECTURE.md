# RagZen Architecture Specification

## Overview

RagZen is an enterprise-grade, local-first RAG framework designed for Python. It provides multi-tenant, permission-aware document retrieval and generation.

```
                    ┌─────────────────────────┐
                    │      Public API         │
                    │ RagZen.local() / ask()  │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ Ingestion Flow   │            │   Query Flow     │
       └────────┬─────────┘            └────────┬─────────┘
                │                               │
       ┌────────▼─────────┐            ┌────────▼─────────┐
       │ Loaders & Check  │            │ Security Context │
       └────────┬─────────┘            └────────┬─────────┘
                │                               │
       ┌────────▼─────────┐            ┌────────▼─────────┐
       │ Chunkers         │            │ Mandatory Filter │
       └────────┬─────────┘            └────────┬─────────┘
                │                               │
       ┌────────▼─────────┐            ┌────────▼─────────┐
       │ Embeddings       │            │ Hybrid Retrieval │
       └────────┬─────────┘            │ (Dense + BM25)   │
                │                      └────────┬─────────┘
       ┌────────▼─────────┐                     │
       │ Vector & Registry│            ┌────────▼─────────┐
       │ SQLite Persistence│           │ LLM Generator    │
       └──────────────────┘            │ & Citation Check │
                                       └──────────────────┘
```

## Key Architectural Principles

1. **Local-First & Multi-Tenant**: Runs without cloud dependencies while strictly enforcing tenant boundary isolation at storage level.
2. **Permission-Aware Retrieval**: Security filters (RBAC/ABAC) are injected into vector and BM25 searches before results are retrieved — preventing cross-tenant data exposure.
3. **Fault-Tolerant & Observability**: Circuit breakers, provider fallback chains, structured logging, and health checks built-in.
4. **Plugin System**: Standard Python protocols for vector stores, embeddings, LLMs, rerankers, and chunkers.
