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

_PROTECTED_FILTER_KEYS = {
    "tenant_id",
    "departments",
    "roles",
    "groups",
    "permissions",
    "owner_id",
    "access_level",
}


def build_mandatory_filters(
    security_context: SecurityContext | None,
    *,
    additional_filters: dict[str, Any] | None = None,
    require_context: bool = True,
    abac_keys: list[str] | None = None,
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

        filters["_security_user_id"] = security_context.user_id
        filters["_security_departments"] = security_context.departments
        filters["_security_roles"] = security_context.roles
        filters["_security_groups"] = security_context.groups
        filters["_security_permissions"] = security_context.permissions
        filters["_security_attributes"] = security_context.attributes
        filters["_security_attribute_keys"] = abac_keys or []

    elif require_context:
        # No context and it's required — this should have been caught earlier
        # but we add an impossible filter as a safety net (fail closed)
        logger.warning("No security context provided but filters required. Applying deny-all.")
        filters["tenant_id"] = "__DENY_ALL__"

    # Merge additional user filters (cannot override security filters)
    if additional_filters:
        for key, value in additional_filters.items():
            if key.startswith("_security") or key in _PROTECTED_FILTER_KEYS:
                logger.warning(
                    "User filter attempted to override security filter: %s (ignored)",
                    key,
                )
                continue
            filters[key] = value

    return filters


def metadata_matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Evaluate ordinary metadata filters and RagZen's mandatory ACL filters."""
    expected_tenant = filters.get("tenant_id")
    if expected_tenant is not None and metadata.get("tenant_id") != expected_tenant:
        return False

    user_id = str(filters.get("_security_user_id", ""))
    owner_id = str(metadata.get("owner_id", ""))
    permissions = set(filters.get("_security_permissions", []))
    owner_or_admin = bool(owner_id and owner_id == user_id) or "*" in permissions

    if not owner_or_admin:
        acl_pairs = (
            ("departments", "_security_departments"),
            ("roles", "_security_roles"),
            ("groups", "_security_groups"),
            ("permissions", "_security_permissions"),
        )
        for metadata_key, filter_key in acl_pairs:
            required = set(metadata.get(metadata_key, []))
            granted = set(filters.get(filter_key, []))
            if metadata_key == "departments" and "all" in granted:
                continue
            if required and not (required & granted):
                return False
        required_attributes = metadata.get("attributes", {})
        granted_attributes = filters.get("_security_attributes", {})
        if any(granted_attributes.get(key) != value for key, value in required_attributes.items()):
            return False

    for key, expected in filters.items():
        if key == "tenant_id" or key.startswith("_security"):
            continue
        actual = metadata.get(key)
        if actual is None:
            return False
        if isinstance(expected, list):
            if isinstance(actual, list):
                if not set(expected) & set(actual):
                    return False
            elif actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


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
