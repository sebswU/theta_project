"""Scene graph structure and extension points.

Defines central scene representation categories without algorithmic logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HumanNode:
    """Human entity container for scene-level state."""

    human_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectNode:
    """Object entity container for scene-level state."""

    object_id: str
    object_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MeshNode:
    """Mesh entity container for scene-level state."""

    mesh_id: str
    vertex_count: int = 0
    face_count: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CameraNode:
    """Camera entity container for scene-level state."""

    camera_id: str
    width: int
    height: int
    intrinsics: dict[str, float] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SensorNode:
    """Sensor entity container for scene-level state."""

    sensor_id: str
    sensor_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RelationshipEdge:
    """Directed relationship edge between two scene entities."""

    source_id: str
    target_id: str
    relation_type: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Scene:
    """Central scene container.

    Scene
     ├── Humans
     ├── Objects
     ├── Meshes
     ├── Cameras
     ├── Sensors
     └── Relationships
    """

    humans: list[HumanNode] = field(default_factory=list)
    objects: list[ObjectNode] = field(default_factory=list)
    meshes: list[MeshNode] = field(default_factory=list)
    cameras: list[CameraNode] = field(default_factory=list)
    sensors: list[SensorNode] = field(default_factory=list)
    relationships: list[RelationshipEdge] = field(default_factory=list)

    # TODO: Add scene versioning and event sourcing metadata.


__all__ = [
    "CameraNode",
    "HumanNode",
    "MeshNode",
    "ObjectNode",
    "RelationshipEdge",
    "Scene",
    "SensorNode",
]
