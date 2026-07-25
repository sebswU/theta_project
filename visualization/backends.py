"""Visualization backend interfaces.

Backends are intentionally defined as interfaces to support Open3D, Blender,
and web dashboard integrations without coupling to specific runtimes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from schemas.models import SceneGraph


class VisualizationBackend(ABC):
    """Universal visualization backend contract."""

    @abstractmethod
    def render(self, scene_graph: SceneGraph) -> dict[str, Any]:
        """Render a scene graph in the target backend.

        Current scaffold returns a stable scene summary that can be consumed by
        tests and higher-level orchestration while concrete backend integrations
        are still being implemented.
        """


def summarize_scene(scene_graph: SceneGraph) -> dict[str, Any]:
    """Create a deterministic summary for a canonical scene graph payload."""

    return {
        "scene_id": scene_graph.scene_id,
        "humans": len(scene_graph.humans),
        "objects": len(scene_graph.objects),
        "meshes": len(scene_graph.meshes),
        "cameras": len(scene_graph.cameras),
        "sensors": len(scene_graph.sensors),
        "relationships": len(scene_graph.relationships),
        "is_empty": not any(
            (
                scene_graph.humans,
                scene_graph.objects,
                scene_graph.meshes,
                scene_graph.cameras,
                scene_graph.sensors,
                scene_graph.relationships,
            )
        ),
    }


class Open3DBackend(VisualizationBackend):
    """Open3D visualization adapter interface."""

    def render(self, scene_graph: SceneGraph) -> dict[str, Any]:
        result = summarize_scene(scene_graph)
        result["backend"] = "open3d"
        return result


class BlenderBackend(VisualizationBackend):
    """Blender visualization adapter interface."""

    def render(self, scene_graph: SceneGraph) -> dict[str, Any]:
        result = summarize_scene(scene_graph)
        result["backend"] = "blender"
        return result


class WebDashboardBackend(VisualizationBackend):
    """Web dashboard visualization adapter interface."""

    def render(self, scene_graph: SceneGraph) -> dict[str, Any]:
        result = summarize_scene(scene_graph)
        result["backend"] = "web_dashboard"
        return result


BACKENDS: dict[str, type[VisualizationBackend]] = {
    "open3d": Open3DBackend,
    "blender": BlenderBackend,
    "web_dashboard": WebDashboardBackend,
}


def get_backend(name: str) -> VisualizationBackend:
    """Resolve and instantiate a visualization backend by name."""

    backend_cls = BACKENDS.get(name)
    if backend_cls is None:
        supported = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unsupported visualization backend '{name}'. Supported backends: {supported}."
        )
    return backend_cls()


__all__ = [
    "BACKENDS",
    "BlenderBackend",
    "Open3DBackend",
    "VisualizationBackend",
    "WebDashboardBackend",
    "get_backend",
    "summarize_scene",
]
