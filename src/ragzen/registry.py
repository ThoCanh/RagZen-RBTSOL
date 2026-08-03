"""Plugin registry with entry-point discovery.

Provides a type-safe registry for RagZen components. Plugins can register
themselves via Python entry points (group: ``ragzen.plugins``) or
programmatically via the registry API.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("ragzen.registry")


@runtime_checkable
class Plugin(Protocol):
    """Protocol that all RagZen plugins should implement."""

    @property
    def name(self) -> str:
        """Unique plugin name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version string."""
        ...

    def health_check(self) -> bool:
        """Return True if the plugin is healthy."""
        ...


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""

    name: str
    version: str
    capability: str
    plugin_class: type[Any]
    instance: Any | None = None
    config_schema: type[Any] | None = None
    compatibility_range: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Central registry for RagZen plugin components.

    Supports both programmatic registration and entry-point discovery.
    Plugins are organized by capability (e.g., 'embedding', 'vector_store',
    'llm', 'chunker', etc.).
    """

    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, PluginInfo]] = {}

    def register(
        self,
        capability: str,
        name: str,
        plugin_class: type[Any],
        *,
        version: str = "0.0.0",
        config_schema: type[Any] | None = None,
        compatibility_range: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a plugin for a given capability.

        Args:
            capability: The capability category (e.g., 'embedding', 'vector_store').
            name: Unique name within the capability.
            plugin_class: The plugin class to register.
            version: Plugin version string.
            config_schema: Optional Pydantic config schema class.
            compatibility_range: Compatible RagZen version range.
            metadata: Additional metadata.
        """
        if capability not in self._plugins:
            self._plugins[capability] = {}

        info = PluginInfo(
            name=name,
            version=version,
            capability=capability,
            plugin_class=plugin_class,
            config_schema=config_schema,
            compatibility_range=compatibility_range,
            metadata=metadata or {},
        )
        self._plugins[capability][name] = info
        logger.debug("Registered plugin: %s/%s v%s", capability, name, version)

    def get(self, capability: str, name: str) -> PluginInfo | None:
        """Look up a registered plugin by capability and name."""
        return self._plugins.get(capability, {}).get(name)

    def list_capability(self, capability: str) -> list[PluginInfo]:
        """List all plugins for a given capability."""
        return list(self._plugins.get(capability, {}).values())

    def list_all(self) -> dict[str, list[PluginInfo]]:
        """List all registered plugins organized by capability."""
        return {cap: list(plugins.values()) for cap, plugins in self._plugins.items()}

    def discover_entry_points(self, group: str = "ragzen.plugins") -> int:
        """Discover and register plugins from Python entry points.

        Returns:
            Number of plugins discovered.
        """
        discovered = 0
        try:
            eps = importlib.metadata.entry_points()
            plugin_eps = (
                eps.select(group=group)
                if hasattr(eps, "select")
                else getattr(eps, "get", lambda g, d: d)(group, [])
            )

            for ep in plugin_eps:
                try:
                    plugin_obj = ep.load()
                    # If it's a class, register it
                    if isinstance(plugin_obj, type):
                        # Try to extract metadata
                        name = getattr(plugin_obj, "plugin_name", ep.name)
                        version = getattr(plugin_obj, "plugin_version", "0.0.0")
                        capability = getattr(plugin_obj, "plugin_capability", "unknown")

                        self.register(
                            capability=capability,
                            name=name,
                            plugin_class=plugin_obj,
                            version=version,
                        )
                        discovered += 1
                        logger.info(
                            "Discovered plugin from entry point: %s (%s/%s)",
                            ep.name,
                            capability,
                            name,
                        )
                except Exception:
                    logger.exception("Failed to load plugin entry point: %s", ep.name)
        except Exception:
            logger.exception("Failed to discover entry points for group: %s", group)

        return discovered

    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()


# Global plugin registry instance
_global_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _global_registry


def reset_registry() -> None:
    """Reset the global plugin registry (for testing)."""
    _global_registry.clear()
