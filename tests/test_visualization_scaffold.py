"""Tests for visualization scaffold behavior."""

import pytest

from schemas import (
    CameraInfo,
    SceneGraph,
    SceneObject,
    SceneRelationship,
    SensorType,
    SourceDescriptor,
)
from visualization import render_scene


def test_render_scene_accepts_canonical_scene_graph() -> None:
    """Public rendering entry point should accept canonical SceneGraph objects."""
    scene = SceneGraph(scene_id="scene-1")

    result = render_scene(scene, backend="open3d")

    assert result["backend"] == "open3d"
    assert result["scene_id"] == "scene-1"
    assert result["is_empty"] is True


def test_render_scene_handles_empty_and_populated_scenes() -> None:
    """Rendering should produce stable summaries for empty and populated scenes."""
    empty_scene = SceneGraph(scene_id="empty")
    populated_scene = SceneGraph(
        scene_id="populated",
        humans=[SceneObject(object_id="human-1", object_type="person")],
        objects=[SceneObject(object_id="object-1", object_type="chair")],
        cameras=[CameraInfo(camera_id="cam-1", width=1920, height=1080)],
        sensors=[SourceDescriptor(source_id="cam-1", source_type=SensorType.RGB_CAMERA)],
        relationships=[
            SceneRelationship(
                source_id="human-1",
                target_id="object-1",
                relation_type="near",
            )
        ],
    )

    empty_result = render_scene(empty_scene, backend="blender")
    populated_result = render_scene(populated_scene, backend="web_dashboard")

    assert empty_result["backend"] == "blender"
    assert empty_result["is_empty"] is True
    assert empty_result["humans"] == 0

    assert populated_result["backend"] == "web_dashboard"
    assert populated_result["is_empty"] is False
    assert populated_result["humans"] == 1
    assert populated_result["objects"] == 1
    assert populated_result["cameras"] == 1
    assert populated_result["sensors"] == 1
    assert populated_result["relationships"] == 1


def test_render_scene_fails_clearly_for_unsupported_backend() -> None:
    """Unsupported backend names should raise a clear and actionable error."""
    with pytest.raises(ValueError, match="Unsupported visualization backend"):
        render_scene(SceneGraph(scene_id="scene-2"), backend="unknown")


def test_render_scene_fails_clearly_for_invalid_payload() -> None:
    """Invalid scene payloads should fail clearly at the public entry point."""
    with pytest.raises(ValueError, match="Invalid canonical scene graph payload"):
        render_scene({"scene": "missing scene_id"}, backend="open3d")

    with pytest.raises(TypeError, match="scene_graph must be"):
        render_scene(SceneGraph(scene_id="scene-1"), backend="open3d")


def test_open3d_backend_collects_keypoints_and_point_clouds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open3D backend should collect geometry payloads without opening a window by default."""
    monkeypatch.setenv("UCA_OPEN3D_SHOW_WINDOW", "0")
    scene = SceneGraph(
        scene_id="scene-open3d-geometry",
        humans=[
            SceneObject(
                object_id="human-1",
                object_type="person",
                attributes={"keypoints_2d": [[10.0, 20.0], [30.0, 40.0]]},
            )
        ],
        objects=[
            SceneObject(
                object_id="pc-1",
                object_type="point_cloud",
                attributes={"points_xyz": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]},
            )
        ],
    )

    result = render_scene(scene, backend="open3d")

    assert result["backend"] == "open3d"
    assert result["keypoint_clouds"] == 1
    assert result["point_clouds"] == 1
    assert result["window_opened"] is False
    assert "rendered" in result
