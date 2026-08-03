"""Mandatory security filters for retrieval queries.

These filters are applied at the storage/retrieval layer to ensure
tenant isolation and permission enforcement. Data is NEVER fetched
across tenants and then filtered in Python.
"""

from __future__ import annotations

import logging
from typing import Any

from ragzen.models import SecurityContext

logger = logging.getLogger("ragzen.security.filters")


def build_mandatory_filters(
    security_context: SecurityContext | None,
    *,
    additional_filters: dict[str, Any] | None = None,
    require_context: bool = True,
) -> dict[str, Any]:
    """Build mandatory filters from a security context.

    These filters MUST be applied to every retrieval query to enforce
    tenant isolation and access control.

    Args:
        security_context: The caller's security context.
        additional_filters: User-specified filters to merge.
        require_context: Whether SecurityContext is required.

    Returns:
        Combined filter dictionary.
    """
    filters: dict[str, Any] = {}

    if security_context is not None:
        # Tenant filter is always mandatory
        filters["tenant_id"] = security_context.tenant_id

        # Department filter (if user has specific departments, not "all")
        if security_context.departments and "all" not in security_context.departments:
            filters["departments"] = security_context.departments

        # Access level filter based on roles/permissions
        # The vector store adapter is responsible for interpreting these
        if security_context.roles:
            filters["_security_roles"] = security_context.roles

        if security_context.groups:
            filters["_security_groups"] = security_context.groups

    elif require_context:
        # No context and it's required — this should have been caught earlier
        # but we add an impossible filter as a safety net (fail closed)
        logger.warning("No security context provided but filters required. Applying deny-all.")
        filters["tenant_id"] = "__DENY_ALL__"

    # Merge additional user filters (cannot override security filters)
    if additional_filters:
        for key, value in additional_filters.items():
            if key in filters and key.startswith(("tenant_id", "_security")):
                logger.warning(
                    "User filter attempted to override security filter: %s (ignored)",
                    key,
                )
                continue
            filters[key] = value

    return filters


def validate_filters_contain_tenant(
    filters: dict[str, Any],
    expected_tenant_id: str | None = None,
) -> bool:
    """Validate that filters contain a tenant_id filter.

    This is a safety check to ensure tenant isolation is enforced.

    Args:
        filters: The filter dictionary to validate.
        expected_tenant_id: If provided, also validate the tenant matches.

    Returns:
        True if the filters are valid.
    """
    if "tenant_id" not in filters:
        return False
    return not (expected_tenant_id and filters["tenant_id"] != expected_tenant_id)
