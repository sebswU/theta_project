"""Visualization backend interfaces.

Backends are intentionally defined as interfaces to support Open3D, Blender,
and web dashboard integrations without coupling to specific runtimes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import os
from typing import Any

import numpy as np

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
        keypoint_clouds = _collect_keypoint_clouds(scene_graph)
        point_clouds = _collect_point_clouds(scene_graph)

        result["keypoint_clouds"] = len(keypoint_clouds)
        result["point_clouds"] = len(point_clouds)

        if not keypoint_clouds and not point_clouds:
            result["rendered"] = False
            result["window_opened"] = False
            result["render_reason"] = "no geometries found"
            return result

        try:
            o3d = importlib.import_module("open3d")
        except ModuleNotFoundError:
            result["rendered"] = False
            result["window_opened"] = False
            result["runtime_unavailable"] = "open3d is not installed"
            return result

        try:
            geometries = _build_open3d_geometries(o3d, keypoint_clouds, point_clouds)

            should_open_window = _should_open_window()
            result["window_opened"] = False
            if should_open_window:
                visualizer = o3d.visualization.Visualizer()
                visualizer.create_window(
                    window_name="Universal CV Adapter - Open3D",
                    width=1280,
                    height=720,
                    visible=True,
                )
                for geometry in geometries:
                    visualizer.add_geometry(geometry)
                visualizer.run()
                visualizer.destroy_window()
                result["window_opened"] = True

            result["rendered"] = True
            result["geometries"] = len(geometries)
            if not should_open_window:
                result["render_reason"] = "window disabled; set UCA_OPEN3D_SHOW_WINDOW=1"
            return result
        except Exception as exc:
            result["rendered"] = False
            result["window_opened"] = False
            result["render_error"] = str(exc)
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


def _collect_keypoint_clouds(scene_graph: SceneGraph) -> list[np.ndarray]:
    """Collect per-human keypoints and map 2D points to z=0 clouds."""
    clouds: list[np.ndarray] = []
    for human in scene_graph.humans:
        keypoints_3d = _coerce_xyz_points(human.attributes.get("keypoints_3d"))
        if keypoints_3d.size:
            clouds.append(keypoints_3d)
            continue

        keypoints_2d = _coerce_xy_points(human.attributes.get("keypoints_2d"))
        if keypoints_2d.size:
            z_col = np.zeros((keypoints_2d.shape[0], 1), dtype=np.float64)
            clouds.append(np.hstack((keypoints_2d, z_col)))
    return clouds


def _collect_point_clouds(scene_graph: SceneGraph) -> list[np.ndarray]:
    """Collect point cloud objects from scene graph attributes."""
    clouds: list[np.ndarray] = []
    for obj in scene_graph.objects:
        points = _coerce_xyz_points(
            obj.attributes.get("points_xyz", obj.attributes.get("point_cloud"))
        )
        if points.size:
            clouds.append(points)
    return clouds


def _coerce_xy_points(raw_points: Any) -> np.ndarray:
    """Convert arbitrary xy keypoint-like payloads into Nx2 float arrays."""
    if raw_points is None:
        return np.empty((0, 2), dtype=np.float64)
    if hasattr(raw_points, "tolist"):
        raw_points = raw_points.tolist()
    if not isinstance(raw_points, list):
        return np.empty((0, 2), dtype=np.float64)

    points: list[list[float]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(points, dtype=np.float64)


def _coerce_xyz_points(raw_points: Any) -> np.ndarray:
    """Convert arbitrary xyz point-like payloads into Nx3 float arrays."""
    if raw_points is None:
        return np.empty((0, 3), dtype=np.float64)
    if hasattr(raw_points, "tolist"):
        raw_points = raw_points.tolist()
    if not isinstance(raw_points, list):
        return np.empty((0, 3), dtype=np.float64)

    points: list[list[float]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            points.append([float(item[0]), float(item[1]), float(item[2])])
        except (TypeError, ValueError):
            continue
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(points, dtype=np.float64)


def _build_open3d_geometries(
    o3d: Any,
    keypoint_clouds: list[np.ndarray],
    point_clouds: list[np.ndarray],
) -> list[Any]:
    """Construct Open3D geometries for keypoints and point clouds."""
    geometries: list[Any] = []

    for points in keypoint_clouds:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        keypoint_color = np.tile(np.array([[0.95, 0.25, 0.2]], dtype=np.float64), (points.shape[0], 1))
        cloud.colors = o3d.utility.Vector3dVector(keypoint_color)
        geometries.append(cloud)

    for points in point_clouds:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.paint_uniform_color([0.15, 0.5, 0.95])
        geometries.append(cloud)

    return geometries


def _should_open_window() -> bool:
    """Return whether Open3D should launch an interactive window."""
    raw_value = os.getenv("UCA_OPEN3D_SHOW_WINDOW", "0").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


__all__ = [
    "BACKENDS",
    "BlenderBackend",
    "Open3DBackend",
    "VisualizationBackend",
    "WebDashboardBackend",
    "get_backend",
    "summarize_scene",
]
