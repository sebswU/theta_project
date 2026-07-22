"""Core architecture interfaces for Universal-CV-Adapter.

This module declares high-level contracts for orchestrating discovery, planning,
scene graph management, and visualization in a model-agnostic way.

Dependency notes:
- Uses Python abstract base classes for explicit interface contracts.
- Uses shared schema models from the `schemas` package.

Extension guidance:
- New adapters and plugins should depend on these contracts instead of concrete classes.
- Keep methods side-effect conscious and typed for agent-friendly implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from schemas.capabilities import SensorCapabilityProfile
from schemas.models import Frame, SceneGraph


class DiscoveryProvider(ABC):
    """Discovers available data sources (hardware, streams, datasets, middleware)."""

    @abstractmethod
    def discover(self) -> list[dict[str, Any]]:
        """Discover source descriptors.

        TODO: Implement concrete environment-specific discovery behavior.
        """


class CapabilityDetector(ABC):
    """Detects typed capabilities from discovered source descriptors."""

    @abstractmethod
    def detect(self, source_descriptor: dict[str, Any]) -> SensorCapabilityProfile:
        """Return normalized capability profile for a source.

        TODO: Implement capability extraction and validation.
        """


class PipelinePlanner(ABC):
    """Plans model and fusion strategy from capabilities and target outputs."""

    @abstractmethod
    def plan(self, capabilities: list[SensorCapabilityProfile]) -> dict[str, Any]:
        """Build a pipeline plan from capability profiles.

        TODO: Implement planning policy and scoring.
        """


class WorkflowBuilder(ABC):
    """Builds executable workflow representations from a planning output."""

    @abstractmethod
    def build(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Build workflow graph/specification from plan.

        TODO: Implement workflow graph compilation.
        """


class SceneGraphManager(ABC):
    """Manages central scene state and relationships across modalities."""

    @abstractmethod
    def update(self, frame_batch: Iterable[Frame]) -> SceneGraph:
        """Update the scene graph using synchronized inputs.

        TODO: Implement scene graph state management and versioning.
        """


class VisualizationBackend(ABC):
    """Defines visualization adapter behavior for rendering scene outputs."""

    @abstractmethod
    def render(self, scene_graph: SceneGraph) -> None:
        """Render scene graph data for a target backend.

        TODO: Implement backend-specific rendering integration.
        """
