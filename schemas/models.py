"""Universal schema models for cross-model interoperability.

This module centralizes exchange schemas consumed/produced by adapters,
model plugins, fusion plugins, and visualization backends.

Implementation notes:
- Pydantic models provide validation and typed contracts.
- Fields are intentionally minimal and extensible for future model outputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CameraInfo(BaseModel):
    """Camera metadata and intrinsic parameters."""

    camera_id: str
    width: int
    height: int
    intrinsics: dict[str, float] = Field(default_factory=dict)


class Calibration(BaseModel):
    """Calibration metadata for extrinsics and quality annotations."""

    calibration_id: str
    reference_frame: str
    extrinsics: list[list[float]] = Field(default_factory=list)


class Frame(BaseModel):
    """Frame payload envelope for image/depth/metadata streams."""

    frame_id: str
    timestamp_ns: int
    source_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Pose2D(BaseModel):
    """2D pose keypoint container."""

    keypoints: list[list[float]] = Field(default_factory=list)


class Pose3D(BaseModel):
    """3D pose keypoint container."""

    keypoints: list[list[float]] = Field(default_factory=list)


class Mask(BaseModel):
    """Binary/instance mask descriptor."""

    mask_id: str
    encoding: str = "rle"


class BoundingBox(BaseModel):
    """2D bounding box representation."""

    x: float
    y: float
    width: float
    height: float
    label: str | None = None


class PointCloud(BaseModel):
    """Point cloud container schema."""

    point_count: int = 0
    coordinate_frame: str = "world"


class Mesh(BaseModel):
    """Mesh data descriptor."""

    mesh_id: str
    vertex_count: int = 0
    face_count: int = 0


class Trajectory(BaseModel):
    """Temporal trajectory schema."""

    trajectory_id: str
    points: list[list[float]] = Field(default_factory=list)


class SceneObject(BaseModel):
    """Generic scene object for graph-based fusion outputs."""

    object_id: str
    object_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class SceneGraph(BaseModel):
    """Central scene representation.

    Scene
     ├── Humans
     ├── Objects
     ├── Meshes
     ├── Cameras
     ├── Sensors
     └── Relationships
    """

    scene_id: str
    humans: list[SceneObject] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    meshes: list[Mesh] = Field(default_factory=list)
    cameras: list[CameraInfo] = Field(default_factory=list)
    sensors: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
