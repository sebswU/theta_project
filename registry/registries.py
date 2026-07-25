"""Registry scaffolding for models and fusion plugins.

Provides registration and lookup contracts for dynamic plugin discovery.
No advanced dependency resolution implementation is included.
"""

from __future__ import annotations

from typing import Any

from fusion.base import FusionPlugin
from registry.cvpr_model import CVPRModel
from schemas.models import RegistryDependency, RegistryResolution


class ModelRegistry:
    """Registry for CVPR model adapters."""

    def __init__(self) -> None:
        self._models: dict[str, type[CVPRModel]] = {}

    def register(self, name: str, model_cls: type[CVPRModel]) -> None:
        """Register a model adapter class.

        TODO: Extend validation and versioned registration.
        """
        if not name.strip():
            raise ValueError("Model name must be non-empty")
        if not issubclass(model_cls, CVPRModel):
            raise TypeError("Registered model must inherit CVPRModel")
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")
        self._models[name] = model_cls

    def get(self, name: str) -> type[CVPRModel]:
        """Return a registered model adapter class by name."""
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model '{name}'") from exc

    def create(self, name: str) -> CVPRModel:
        """Instantiate a registered model adapter by name."""
        return self.get(name)()

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
        model = self.create(name)
        requirements = model.get_requirements()
        dependencies_raw = requirements.resources.get("dependencies", [])

        dependencies: list[RegistryDependency] = []
        for dep in dependencies_raw:
            dependency = _coerce_dependency(dep)
            if dependency is not None:
                dependencies.append(dependency)

        return RegistryResolution(
            name=name,
            dependencies=dependencies,
            metadata={"minimum_version": requirements.minimum_version},
        )


class PluginRegistry:
    """Registry for fusion plugin adapters."""

    def __init__(self) -> None:
        self._plugins: dict[str, type[FusionPlugin]] = {}

    def register(self, name: str, plugin_cls: type[FusionPlugin]) -> None:
        """Register a fusion plugin class.

        TODO: Extend validation and lifecycle metadata.
        """
        if not name.strip():
            raise ValueError("Plugin name must be non-empty")
        if not issubclass(plugin_cls, FusionPlugin):
            raise TypeError("Registered plugin must inherit FusionPlugin")
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' is already registered")
        self._plugins[name] = plugin_cls

    def get(self, name: str) -> type[FusionPlugin]:
        """Return a registered fusion plugin class by name."""
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise KeyError(f"Unknown plugin '{name}'") from exc

    def create(self, name: str) -> FusionPlugin:
        """Instantiate a registered fusion plugin by name."""
        return self.get(name)()

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
        if name not in self._plugins:
            raise KeyError(f"Unknown plugin '{name}'")
        return RegistryResolution(name=name)


def _coerce_dependency(dep: Any) -> RegistryDependency | None:
    """Convert common dependency representations to typed entries."""
    if isinstance(dep, RegistryDependency):
        return dep
    if isinstance(dep, str):
        package_name, version_spec = _split_dependency_string(dep)
        return RegistryDependency(package_name=package_name, version_spec=version_spec)
    if isinstance(dep, dict):
        package_name = dep.get("package_name")
        if not isinstance(package_name, str) or not package_name.strip():
            return None
        version_spec = dep.get("version_spec")
        optional = bool(dep.get("optional", False))
        metadata = dep.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return RegistryDependency(
            package_name=package_name,
            version_spec=version_spec if isinstance(version_spec, str) else None,
            optional=optional,
            metadata=metadata,
        )
    return None


def _split_dependency_string(spec: str) -> tuple[str, str | None]:
    """Split compact specs like 'numpy>=1.26' into package/version fields."""
    operators = ("==", ">=", "<=", "~=", "!=", ">", "<")
    for operator in operators:
        if operator in spec:
            package_name, version = spec.split(operator, maxsplit=1)
            return package_name.strip(), f"{operator}{version.strip()}"
    return spec.strip(), None
