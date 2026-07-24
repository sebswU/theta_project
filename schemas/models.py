"""Universal schema models for cross-model interoperability.

This module centralizes exchange schemas consumed/produced by adapters,
model plugins, fusion plugins, and visualization backends.

Implementation notes:
- Pydantic models provide validation and typed contracts.
- Fields are intentionally minimal and extensible for future model outputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from schemas.capabilities import SensorCapabilityProfile, SensorType


class CameraInfo(BaseModel):
    """Camera metadata and intrinsic parameters."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    width: int
    height: int
    intrinsics: dict[str, float] = Field(default_factory=dict)


class Calibration(BaseModel):
    """Calibration metadata for extrinsics and quality annotations."""

    model_config = ConfigDict(extra="forbid")

    calibration_id: str
    reference_frame: str
    extrinsics: list[list[float]] = Field(default_factory=list)


class Frame(BaseModel):
    """Frame payload envelope for image/depth/metadata streams."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str
    timestamp_ns: int
    source_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Pose2D(BaseModel):
    """2D pose keypoint container."""

    model_config = ConfigDict(extra="forbid")

    keypoints: list[list[float]] = Field(default_factory=list)


class Pose3D(BaseModel):
    """3D pose keypoint container."""

    model_config = ConfigDict(extra="forbid")

    keypoints: list[list[float]] = Field(default_factory=list)


class Mask(BaseModel):
    """Binary/instance mask descriptor."""

    model_config = ConfigDict(extra="forbid")

    mask_id: str
    encoding: str = "rle"


class BoundingBox(BaseModel):
    """2D bounding box representation."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    width: float
    height: float
    label: str | None = None


class PointCloud(BaseModel):
    """Point cloud container schema."""

    model_config = ConfigDict(extra="forbid")

    point_count: int = 0
    coordinate_frame: str = "world"


class Mesh(BaseModel):
    """Mesh data descriptor."""

    model_config = ConfigDict(extra="forbid")

    mesh_id: str
    vertex_count: int = 0
    face_count: int = 0


class Trajectory(BaseModel):
    """Temporal trajectory schema."""

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    points: list[list[float]] = Field(default_factory=list)


class SceneObject(BaseModel):
    """Generic scene object for graph-based fusion outputs."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    object_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class SceneRelationship(BaseModel):
    """Relationship edge between scene entities."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    relation_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class SourceDescriptor(BaseModel):
    """Discovery output describing a source available to the system."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SensorType
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelinePlan(BaseModel):
    """Typed orchestration plan spanning discovery, registry, and fusion."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    capabilities: list[SensorCapabilityProfile] = Field(default_factory=list)
    selected_models: list[str] = Field(default_factory=list)
    selected_plugins: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class WorkflowNode(BaseModel):
    """Workflow graph node definition."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: str
    config: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    """Workflow graph edge definition."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    relation_type: str = "depends_on"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowGraph(BaseModel):
    """Execution graph description for orchestrated workflows."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusionConfiguration(BaseModel):
    """Runtime configuration for fusion plugins."""

    model_config = ConfigDict(extra="forbid")

    plugin_name: str
    version: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class FusionRequest(BaseModel):
    """Normalized fusion input payload."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    inputs: list[Frame] = Field(default_factory=list)
    scene_graph: SceneGraph | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusionResponse(BaseModel):
    """Normalized fusion output payload."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    scene_graph: SceneGraph | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCapabilities(BaseModel):
    """Registry-facing capability summary for model adapters."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    supported_sources: list[SensorType] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelRequirements(BaseModel):
    """Registry-facing runtime requirements for model adapters."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    required_sources: list[SensorType] = Field(default_factory=list)
    minimum_version: str | None = None
    resources: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    """Inference input envelope for registry-managed model adapters."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    frames: list[Frame] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """Inference output envelope for registry-managed model adapters."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    scene_graph: SceneGraph | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputSchema(BaseModel):
    """Describes the public output contract for a model adapter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    fields: list[str] = Field(default_factory=list)
    mime_type: str = "application/json"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistryDependency(BaseModel):
    """Typed runtime dependency entry used by registry resolution."""

    model_config = ConfigDict(extra="forbid")

    package_name: str
    version_spec: str | None = None
    optional: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistryResolution(BaseModel):
    """Resolved runtime dependency bundle for a registered component."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dependencies: list[RegistryDependency] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    humans: list[SceneObject] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    meshes: list[Mesh] = Field(default_factory=list)
    cameras: list[CameraInfo] = Field(default_factory=list)
    sensors: list[SourceDescriptor] = Field(default_factory=list)
    relationships: list[SceneRelationship] = Field(default_factory=list)


__all__ = [
    "BoundingBox",
    "Calibration",
    "CameraInfo",
    "Frame",
    "FusionConfiguration",
    "FusionRequest",
    "FusionResponse",
    "InferenceRequest",
    "InferenceResponse",
    "Mask",
    "Mesh",
    "ModelCapabilities",
    "ModelRequirements",
    "OutputSchema",
    "PipelinePlan",
    "PointCloud",
    "Pose2D",
    "Pose3D",
    "RegistryDependency",
    "RegistryResolution",
    "SceneGraph",
    "SceneObject",
    "SceneRelationship",
    "SourceDescriptor",
    "Trajectory",
    "WorkflowEdge",
    "WorkflowGraph",
    "WorkflowNode",
]
