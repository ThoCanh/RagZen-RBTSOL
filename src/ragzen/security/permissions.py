"""Permission evaluation engine.

Provides the AuthorizationPolicy protocol and a default implementation
that combines RBAC and ABAC checks.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ragzen.models import AccessControl, SecurityContext

logger = logging.getLogger("ragzen.security.permissions")


@runtime_checkable
class AuthorizationPolicy(Protocol):
    """Protocol for authorization policy implementations.

    Plugins can provide custom authorization policies by implementing
    this protocol and registering via the plugin registry.
    """

    def evaluate(
        self,
        security_context: SecurityContext,
        resource_access_control: AccessControl,
        action: str = "read",
    ) -> bool:
        """Evaluate whether the security context grants access.

        Args:
            security_context: The caller's context.
            resource_access_control: The resource's ACL.
            action: The action being performed (read, write, delete).

        Returns:
            True if access is granted.
        """
        ...


class DefaultAuthorizationPolicy:
    """Default policy combining RBAC and ABAC checks.

    Rules (in order):
    1. Tenant must match (mandatory).
    2. Owner always has access.
    3. Wildcard permission ('*') grants full access.
    4. Department, role, group, and permission checks must all pass
       when the corresponding ACL fields are non-empty.
    """

    def __init__(self, *, fail_closed: bool = True) -> None:
        self._fail_closed = fail_closed

    def evaluate(
        self,
        security_context: SecurityContext,
        resource_access_control: AccessControl,
        action: str = "read",
    ) -> bool:
        """Evaluate access using combined RBAC + ABAC rules."""
        # 1. Tenant isolation is mandatory
        if security_context.tenant_id != resource_access_control.tenant_id:
            logger.debug(
                "Tenant mismatch: ctx=%s, resource=%s",
                security_context.tenant_id,
                resource_access_control.tenant_id,
            )
            return False

        # 2. Owner bypass
        if (
            resource_access_control.owner_id
            and security_context.user_id == resource_access_control.owner_id
        ):
            return True

        # 3. Wildcard permission
        if "*" in security_context.permissions:
            return True

        # 4. Department check
        if resource_access_control.departments:
            has_dept = bool(
                set(security_context.departments) & set(resource_access_control.departments)
            ) or "all" in security_context.departments
            if not has_dept:
                return False

        # 5. Role check
        if resource_access_control.roles:
            has_role = bool(
                set(security_context.roles) & set(resource_access_control.roles)
            )
            if not has_role and self._fail_closed:
                    return False

        # 6. Group check
        if resource_access_control.groups:
            has_group = bool(
                set(security_context.groups) & set(resource_access_control.groups)
            )
            if not has_group:
                return False

        # 7. Permission check
        if resource_access_control.permissions:
            has_perm = bool(
                set(security_context.permissions) & set(resource_access_control.permissions)
            )
            if not has_perm:
                return False

        return True
