"""Registry scaffolding for models and fusion plugins.

Provides registration and lookup contracts for dynamic plugin discovery.
No advanced dependency resolution implementation is included.
"""

from __future__ import annotations

from fusion.base import FusionPlugin
from registry.cvpr_model import CVPRModel
from schemas.models import RegistryResolution


class ModelRegistry:
    """Registry for CVPR model adapters."""

    def __init__(self) -> None:
        self._models: dict[str, type[CVPRModel]] = {}

    def register(self, name: str, model_cls: type[CVPRModel]) -> None:
        """Register a model adapter class.

        TODO: Extend validation and versioned registration.
        """
        self._models[name] = model_cls

    def discover(self) -> list[str]:
        """List registered model adapter names.

        TODO: Add filter and metadata support.
        """
        return sorted(self._models.keys())

    def validate(self, name: str) -> bool:
        """Validate registry presence.

        TODO: Add interface conformance checks.
        """
        return name in self._models

    def resolve_dependencies(self, name: str) -> RegistryResolution:
        """Resolve runtime dependency metadata for a model.

        TODO: Integrate dependency graph and environment constraints.
        """
        raise NotImplementedError("TODO: implement dependency resolution")


class PluginRegistry:
    """Registry for fusion plugin adapters."""

    def __init__(self) -> None:
        self._plugins: dict[str, type[FusionPlugin]] = {}

    def register(self, name: str, plugin_cls: type[FusionPlugin]) -> None:
        """Register a fusion plugin class.

        TODO: Extend validation and lifecycle metadata.
        """
        self._plugins[name] = plugin_cls

    def discover(self) -> list[str]:
        """List registered plugin names.

        TODO: Add capability filtering support.
        """
        return sorted(self._plugins.keys())

    def validate(self, name: str) -> bool:
        """Validate plugin presence.

        TODO: Add plugin interface compliance checks.
        """
        return name in self._plugins

    def resolve_dependencies(self, name: str) -> RegistryResolution:
        """Resolve runtime dependency metadata for a plugin.

        TODO: Integrate dependency graph and environment constraints.
        """
        raise NotImplementedError("TODO: implement dependency resolution")
