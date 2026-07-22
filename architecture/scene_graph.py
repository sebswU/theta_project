"""Scene graph structure and extension points.

Defines central scene representation categories without algorithmic logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    humans: list[dict[str, Any]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    meshes: list[dict[str, Any]] = field(default_factory=list)
    cameras: list[dict[str, Any]] = field(default_factory=list)
    sensors: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    # TODO: Add strongly typed node and edge classes.
    # TODO: Add scene versioning and event sourcing metadata.
