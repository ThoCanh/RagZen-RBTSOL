# RagZen Threat Model

## Overview

RagZen processes multi-tenant enterprise data and integrates with vector databases and Large Language Models.
This threat model identifies key assets, threat actors, attack vectors, and implemented mitigations.

## Asset Classification

1. **Enterprise Documents & Embeddings** (Confidentiality & Integrity)
2. **Security Context & ACLs** (Integrity & Confidentiality)
3. **LLM Provider Credentials** (Confidentiality)
4. **Audit Logs** (Integrity & Accountability)

## Threat Actors & Vectors

### 1. Cross-Tenant Data Leakage
- **Threat**: Tenant A attempts to access or search Tenant B's documents.
- **Mitigation**: Mandatory tenant_id filter is injected into ALL queries at the storage layer before scoring/retrieval. Fail-closed model rejects queries without valid tenant scope.

### 2. Prompt Injection (Direct & Indirect)
- **Threat**: Malicious document content or user query attempts to override system prompt rules (e.g. "Ignore previous instructions").
- **Mitigation**:
  - Heuristic `PromptInjectionDetector` scans incoming queries and documents.
  - Strict system prompt instructing model to rely strictly on context.
  - Permission filters are enforced BEFORE document content is sent to LLM. Untrusted text never bypasses permission layer.

### 3. Path Traversal & Unsafe File Upload
- **Threat**: File path with `../` attempts to read arbitrary host system files.
- **Mitigation**: `safe_resolve_path()` verifies that resolved file paths lie strictly within allowed root directory. File size and MIME allowlist validation.

### 4. Secret & PII Leakage
- **Threat**: API keys, tokens, or PII exposed in logs or API responses.
- **Mitigation**: `redact_secrets()` and `redact_dict()` strip sensitive patterns prior to logging. Config secrets use `SecretStr`.

### 5. Denial of Service / Resource Exhaustion
- **Threat**: Oversized document ingestion or excessive token consumption.
- **Mitigation**: Configurable file size limits (`max_file_size_mb`), token budgets, timeouts, and batch limits.
