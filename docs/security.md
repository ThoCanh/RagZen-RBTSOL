# Security

`SecurityContext` is constructed from authenticated identity claims. Tenant and ACL
filters are immutable and cannot be replaced by caller metadata filters. The built-in
authorization model covers owners, departments, roles, groups, permissions and
declared ABAC attributes.

Production FastAPI mode requires at least one server-side API principal. Request body
security contexts are only a development convenience when production mode is off.
Filesystem path ingestion is disabled until explicit roots are configured.

See `THREAT_MODEL.md` and `SECURITY.md` in the repository root for reporting and trust
boundary details.
