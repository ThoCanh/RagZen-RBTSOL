"""Tests for plugin registry."""

from __future__ import annotations

from ragzen.registry import PluginRegistry, get_registry, reset_registry


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def setup_method(self) -> None:
        self.registry = PluginRegistry()

    def test_register_and_get(self) -> None:
        class MyEmbedding:
            pass

        self.registry.register("embedding", "test_embedding", MyEmbedding, version="1.0.0")
        info = self.registry.get("embedding", "test_embedding")
        assert info is not None
        assert info.name == "test_embedding"
        assert info.version == "1.0.0"
        assert info.plugin_class is MyEmbedding

    def test_get_nonexistent(self) -> None:
        assert self.registry.get("embedding", "nonexistent") is None
        assert self.registry.get("nonexistent", "anything") is None

    def test_list_capability(self) -> None:
        class A:
            pass

        class B:
            pass

        self.registry.register("embedding", "a", A, version="1.0")
        self.registry.register("embedding", "b", B, version="2.0")
        plugins = self.registry.list_capability("embedding")
        assert len(plugins) == 2
        names = {p.name for p in plugins}
        assert names == {"a", "b"}

    def test_list_empty_capability(self) -> None:
        assert self.registry.list_capability("nonexistent") == []

    def test_list_all(self) -> None:
        class X:
            pass

        self.registry.register("embedding", "e1", X)
        self.registry.register("vector_store", "v1", X)
        all_plugins = self.registry.list_all()
        assert "embedding" in all_plugins
        assert "vector_store" in all_plugins

    def test_overwrite_registration(self) -> None:
        class V1:
            pass

        class V2:
            pass

        self.registry.register("llm", "test", V1, version="1.0")
        self.registry.register("llm", "test", V2, version="2.0")
        info = self.registry.get("llm", "test")
        assert info is not None
        assert info.plugin_class is V2
        assert info.version == "2.0"

    def test_register_with_metadata(self) -> None:
        class Comp:
            pass

        self.registry.register(
            "chunker",
            "test",
            Comp,
            version="1.0",
            compatibility_range=">=0.1.0,<1.0.0",
            metadata={"author": "test"},
        )
        info = self.registry.get("chunker", "test")
        assert info is not None
        assert info.compatibility_range == ">=0.1.0,<1.0.0"
        assert info.metadata == {"author": "test"}

    def test_clear(self) -> None:
        class C:
            pass

        self.registry.register("x", "y", C)
        assert self.registry.get("x", "y") is not None
        self.registry.clear()
        assert self.registry.get("x", "y") is None

    def test_discover_entry_points_returns_count(self) -> None:
        # With no plugins installed, should return 0
        count = self.registry.discover_entry_points(group="ragzen.test.nonexistent")
        assert count == 0


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def setup_method(self) -> None:
        reset_registry()

    def test_get_registry_singleton(self) -> None:
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_registry(self) -> None:
        class T:
            pass

        reg = get_registry()
        reg.register("test", "t", T)
        assert reg.get("test", "t") is not None
        reset_registry()
        assert reg.get("test", "t") is None
