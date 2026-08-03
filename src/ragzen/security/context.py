"""Security context and tenant isolation.

The SecurityContext is the primary mechanism for permission-aware operations.
It is immutable after creation and used to generate mandatory filters
that are applied at the storage/retrieval layer — never post-filtered in Python.
"""

from __future__ import annotations

import logging

from ragzen.exceptions import (
    SecurityContextRequiredError,
    TenantIsolationError,
)
from ragzen.models import AccessControl, SecurityContext

logger = logging.getLogger("ragzen.security")


def require_security_context(
    context: SecurityContext | None,
    *,
    require: bool = True,
) -> SecurityContext | None:
    """Validate that a security context is provided when required.

    Args:
        context: The security context (may be None).
        require: Whether to enforce the requirement.

    Returns:
        The validated SecurityContext, or None if not required.

    Raises:
        SecurityContextRequiredError: If required but not provided.
    """
    if require and context is None:
        raise SecurityContextRequiredError(
            "SecurityContext is required for this operation. "
            "Provide a SecurityContext or configure require_security_context=False "
            "for public collections."
        )
    return context


def validate_tenant_access(
    security_context: SecurityContext,
    resource_tenant_id: str,
) -> None:
    """Validate that the security context has access to the given tenant.

    Args:
        security_context: The caller's security context.
        resource_tenant_id: The tenant ID of the resource being accessed.

    Raises:
        TenantIsolationError: If the security context's tenant doesn't match.
    """
    if security_context.tenant_id != resource_tenant_id:
        logger.warning(
            "Tenant isolation violation: context tenant=%s, resource tenant=%s, user=%s",
            security_context.tenant_id,
            resource_tenant_id,
            security_context.user_id,
        )
        raise TenantIsolationError(
            f"Access denied: tenant '{security_context.tenant_id}' cannot access "
            f"resources belonging to tenant '{resource_tenant_id}'."
        )


def check_access_control(
    security_context: SecurityContext,
    access_control: AccessControl,
    *,
    fail_closed: bool = True,
) -> bool:
    """Check if a security context grants access to a resource with the given ACL.

    Implements a fail-closed model by default: if any check is ambiguous,
    access is denied.

    Args:
        security_context: The caller's security context.
        access_control: The resource's access control metadata.
        fail_closed: If True, deny on any ambiguity.

    Returns:
        True if access is granted, False if denied.
    """
    # Tenant must always match
    if security_context.tenant_id != access_control.tenant_id:
        return False

    # Owner always has access
    if access_control.owner_id and security_context.user_id == access_control.owner_id:
        return True

    # Check wildcard permissions
    if "*" in security_context.permissions:
        return True

    # Check department access
    if access_control.departments:
        has_department = any(
            dept in security_context.departments for dept in access_control.departments
        )
        # "all" department grants access to everything
        has_all = "all" in security_context.departments
        if not has_department and not has_all:
            return False

    # Check role access
    if access_control.roles:
        has_role = any(role in security_context.roles for role in access_control.roles)
        if not has_role and fail_closed:
                return False

    # Check group access
    if access_control.groups:
        has_group = any(group in security_context.groups for group in access_control.groups)
        if not has_group:
            return False

    # Check explicit permissions
    if access_control.permissions:
        has_permission = any(
            perm in security_context.permissions for perm in access_control.permissions
        )
        if not has_permission:
            return False

    return True
