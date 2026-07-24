"""Public schema contracts shared across discovery, orchestration, registry, and fusion."""

from .capabilities import SensorCapabilityProfile
from .capabilities import SensorType
from .models import BoundingBox
from .models import Calibration
from .models import CameraInfo
from .models import Frame
from .models import FusionConfiguration
from .models import FusionRequest
from .models import FusionResponse
from .models import InferenceRequest
from .models import InferenceResponse
from .models import Mask
from .models import Mesh
from .models import ModelCapabilities
from .models import ModelRequirements
from .models import OutputSchema
from .models import PipelinePlan
from .models import PointCloud
from .models import Pose2D
from .models import Pose3D
from .models import RegistryDependency
from .models import RegistryResolution
from .models import SceneGraph
from .models import SceneObject
from .models import SceneRelationship
from .models import SourceDescriptor
from .models import Trajectory
from .models import WorkflowEdge
from .models import WorkflowGraph
from .models import WorkflowNode

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
    "SensorCapabilityProfile",
    "SensorType",
    "SourceDescriptor",
    "Trajectory",
    "WorkflowEdge",
    "WorkflowGraph",
    "WorkflowNode",
]
