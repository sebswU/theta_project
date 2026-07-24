"""Core architecture package for scene graph and orchestration contracts."""

from .core_interfaces import (
    CapabilityDetector,
    DiscoveryProvider,
    PipelinePlanner,
    SceneGraphManager,
    VisualizationBackend,
    WorkflowBuilder,
)
from .scene_graph import (
    CameraNode,
    HumanNode,
    MeshNode,
    ObjectNode,
    RelationshipEdge,
    Scene,
    SensorNode,
)

__all__ = [
    "CapabilityDetector",
    "CameraNode",
    "DiscoveryProvider",
    "HumanNode",
    "MeshNode",
    "ObjectNode",
    "PipelinePlanner",
    "RelationshipEdge",
    "Scene",
    "SceneGraphManager",
    "SensorNode",
    "VisualizationBackend",
    "WorkflowBuilder",
]