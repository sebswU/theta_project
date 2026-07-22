"""Visualization backend interfaces.

Backends are intentionally defined as interfaces to support Open3D, Blender,
and web dashboard integrations without coupling to specific runtimes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.models import SceneGraph


class VisualizationBackend(ABC):
    """Universal visualization backend contract."""

    @abstractmethod
    def render(self, scene_graph: SceneGraph) -> None:
        """Render a scene graph in the target backend.

        TODO: Implement backend rendering integration.
        """


class Open3DBackend(VisualizationBackend):
    """Open3D visualization adapter interface."""


class BlenderBackend(VisualizationBackend):
    """Blender visualization adapter interface."""


class WebDashboardBackend(VisualizationBackend):
    """Web dashboard visualization adapter interface."""
