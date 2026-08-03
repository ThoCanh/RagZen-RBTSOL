# RagZen Security Policy and Vulnerability Disclosure

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Security is a core priority of RagZen. If you discover a security vulnerability,
please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email security findings to `security@ragzen.org` or report via GitHub Security Advisories.
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

We will acknowledge receipt within 24 hours and aim to release a patch within 7 business days for critical vulnerabilities.

## Security Architecture

- **Tenant Isolation**: Mandatory filters enforced at storage and retrieval layers.
- **Fail-Closed Permissions**: Default policy denies access if any permission evaluation is ambiguous.
- **Secret Redaction**: Automatic redaction of API keys, passwords, and tokens in logs.
- **Input Sanitization**: Path traversal checks, file size limits, MIME type allowlists.
- **Prompt Injection Screening**: Heuristic detection of malicious prompt overrides in user queries and documents.
