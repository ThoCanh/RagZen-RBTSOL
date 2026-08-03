"""Cross-tenant security tests.

These tests verify that tenant isolation is enforced at all layers
and that no data leaks between tenants.
"""

from __future__ import annotations

import pytest

from ragzen.exceptions import TenantIsolationError
from ragzen.models import AccessControl, SecurityContext
from ragzen.security.context import check_access_control, validate_tenant_access
from ragzen.security.filters import build_mandatory_filters
from ragzen.security.permissions import DefaultAuthorizationPolicy


class TestCrossTenantIsolation:
    """Tests that tenant isolation cannot be bypassed."""

    def test_tenant_filter_always_applied(self) -> None:
        """Mandatory filters must always include tenant_id."""
        ctx = SecurityContext(tenant_id="company-a", user_id="u1")
        filters = build_mandatory_filters(ctx)
        assert "tenant_id" in filters
        assert filters["tenant_id"] == "company-a"

    def test_user_cannot_override_tenant_filter(self) -> None:
        """User-provided filters must not override tenant isolation."""
        ctx = SecurityContext(tenant_id="company-a", user_id="u1")
        filters = build_mandatory_filters(
            ctx,
            additional_filters={"tenant_id": "company-b"},
        )
        # Must keep the original tenant, not the user override
        assert filters["tenant_id"] == "company-a"

    def test_cross_tenant_access_denied(self) -> None:
        """Access control must deny cross-tenant access."""
        ctx_a = SecurityContext(tenant_id="company-a", user_id="u1", roles=["admin"])
        ac_b = AccessControl(tenant_id="company-b")
        assert check_access_control(ctx_a, ac_b) is False

    def test_cross_tenant_validation_raises(self) -> None:
        """validate_tenant_access must raise on cross-tenant access."""
        ctx = SecurityContext(tenant_id="company-a")
        with pytest.raises(TenantIsolationError):
            validate_tenant_access(ctx, "company-b")

    def test_admin_cannot_access_other_tenant(self) -> None:
        """Even admin users are restricted to their tenant."""
        ctx = SecurityContext(
            tenant_id="company-a",
            user_id="admin",
            roles=["admin"],
            permissions=["*"],
        )
        ac = AccessControl(tenant_id="company-b")
        assert check_access_control(ctx, ac) is False

    def test_wildcard_does_not_bypass_tenant(self) -> None:
        """Wildcard permissions don't cross tenant boundaries."""
        ctx = SecurityContext(tenant_id="t1", permissions=["*"])
        ac = AccessControl(tenant_id="t2", permissions=["*"])
        policy = DefaultAuthorizationPolicy()
        assert policy.evaluate(ctx, ac) is False

    def test_owner_of_other_tenant_denied(self) -> None:
        """Owner in one tenant cannot access owned resource in another tenant."""
        ctx = SecurityContext(tenant_id="company-a", user_id="owner-1")
        ac = AccessControl(tenant_id="company-b", owner_id="owner-1")
        assert check_access_control(ctx, ac) is False

    def test_no_security_context_denied(self) -> None:
        """Without SecurityContext, filters apply deny-all."""
        filters = build_mandatory_filters(None, require_context=True)
        assert filters["tenant_id"] == "__DENY_ALL__"

    def test_empty_tenant_rejected_at_model_level(self) -> None:
        """SecurityContext cannot be created with empty tenant_id."""
        with pytest.raises(Exception, match="tenant_id"):
            SecurityContext(tenant_id="", user_id="u1")

    def test_multiple_tenants_isolated(self) -> None:
        """Each tenant's filters are independent."""
        ctx_a = SecurityContext(tenant_id="a", departments=["eng"])
        ctx_b = SecurityContext(tenant_id="b", departments=["eng"])

        filters_a = build_mandatory_filters(ctx_a)
        filters_b = build_mandatory_filters(ctx_b)

        assert filters_a["tenant_id"] == "a"
        assert filters_b["tenant_id"] == "b"
        assert filters_a["tenant_id"] != filters_b["tenant_id"]


class TestFilterBypass:
    """Tests that security filters cannot be bypassed."""

    def test_security_prefix_protected(self) -> None:
        """User cannot override _security prefixed filters."""
        ctx = SecurityContext(
            tenant_id="t1",
            roles=["viewer"],
            groups=["group-a"],
        )
        filters = build_mandatory_filters(
            ctx,
            additional_filters={
                "_security_roles": ["admin"],
                "_security_groups": ["admin-group"],
            },
        )
        # Security filters should not be overridden by user
        assert filters["_security_roles"] == ["viewer"]
        assert filters["_security_groups"] == ["group-a"]

    def test_malicious_metadata_filter(self) -> None:
        """Malicious metadata values in filters don't affect security."""
        ctx = SecurityContext(tenant_id="t1")
        filters = build_mandatory_filters(
            ctx,
            additional_filters={
                "$gt": {"access_level": "public"},
                "__proto__": "exploit",
                "constructor": "exploit",
            },
        )
        # Security filters preserved
        assert filters["tenant_id"] == "t1"
