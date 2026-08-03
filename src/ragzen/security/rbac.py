"""RBAC (Role-Based Access Control) policy.

Provides a standalone RBAC policy that can be used independently
or composed with other policies.
"""

from __future__ import annotations

import logging

from ragzen.models import AccessControl, SecurityContext

logger = logging.getLogger("ragzen.security.rbac")


class RBACPolicy:
    """Role-Based Access Control policy.

    Checks access based on:
    1. Tenant isolation (mandatory)
    2. Role membership
    3. Owner access
    """

    def __init__(self, *, fail_closed: bool = True) -> None:
        self._fail_closed = fail_closed

    def evaluate(
        self,
        security_context: SecurityContext,
        resource_access_control: AccessControl,
        action: str = "read",
    ) -> bool:
        """Evaluate RBAC access.

        Args:
            security_context: The caller's context.
            resource_access_control: The resource's ACL.
            action: The action (read, write, delete).

        Returns:
            True if access is granted.
        """
        # Tenant must match
        if security_context.tenant_id != resource_access_control.tenant_id:
            return False

        # Owner bypass
        if (
            resource_access_control.owner_id
            and security_context.user_id == resource_access_control.owner_id
        ):
            return True

        # Wildcard
        if "*" in security_context.permissions:
            return True

        # Role check
        if resource_access_control.roles:
            has_role = bool(set(security_context.roles) & set(resource_access_control.roles))
            if not has_role:
                return not self._fail_closed

        return True
