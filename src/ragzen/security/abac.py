"""ABAC (Attribute-Based Access Control) policy.

Provides attribute-based access checks using the security context's
attributes dictionary and the resource's access control metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from ragzen.models import AccessControl, AccessLevel, SecurityContext

logger = logging.getLogger("ragzen.security.abac")

# Access level hierarchy (higher index = more restricted)
_ACCESS_LEVEL_HIERARCHY: dict[AccessLevel, int] = {
    AccessLevel.PUBLIC: 0,
    AccessLevel.INTERNAL: 1,
    AccessLevel.CONFIDENTIAL: 2,
    AccessLevel.RESTRICTED: 3,
}


class ABACPolicy:
    """Attribute-Based Access Control policy.

    Evaluates access based on:
    1. Tenant isolation (mandatory)
    2. Access level clearance
    3. Department membership
    4. Custom attribute matching
    """

    def __init__(
        self,
        *,
        fail_closed: bool = True,
        max_access_level: AccessLevel = AccessLevel.RESTRICTED,
    ) -> None:
        self._fail_closed = fail_closed
        self._max_access_level = max_access_level

    def evaluate(
        self,
        security_context: SecurityContext,
        resource_access_control: AccessControl,
        action: str = "read",
    ) -> bool:
        """Evaluate ABAC access.

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

        # Check access level clearance
        resource_level = resource_access_control.access_level
        user_clearance = self._get_user_clearance(security_context)

        resource_rank = _ACCESS_LEVEL_HIERARCHY.get(resource_level, 0)
        user_rank = _ACCESS_LEVEL_HIERARCHY.get(user_clearance, 0)

        if resource_rank > user_rank:
            logger.debug(
                "Access level denied: resource=%s, user_clearance=%s",
                resource_level,
                user_clearance,
            )
            return False

        # Department check
        if resource_access_control.departments:
            has_dept = bool(
                set(security_context.departments) & set(resource_access_control.departments)
            ) or "all" in security_context.departments
            if not has_dept:
                return False

        return True

    def _get_user_clearance(self, security_context: SecurityContext) -> AccessLevel:
        """Determine user's access level clearance from attributes or roles.

        If the security context has an 'access_level' attribute, use that.
        Otherwise, infer from roles:
        - admin -> RESTRICTED
        - manager -> CONFIDENTIAL
        - default -> INTERNAL
        """
        # Check explicit attribute
        explicit_level: Any = security_context.attributes.get("access_level")
        if explicit_level:
            try:
                return AccessLevel(explicit_level)
            except ValueError:
                pass

        # Infer from roles
        roles_set = set(security_context.roles)
        if "admin" in roles_set or any(r.endswith("_admin") for r in roles_set):
            return AccessLevel.RESTRICTED
        if any("manager" in r for r in roles_set):
            return AccessLevel.CONFIDENTIAL
        return AccessLevel.INTERNAL
