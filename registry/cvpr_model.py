"""Abstract base class contract for CVPR/research model adapters.

All future model integrations must implement this interface with a thin
adapter layer only, keeping model-specific code isolated.

TODO:
- Add runtime lifecycle hooks for warm-start and teardown.
- Add async variants for distributed deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.models import (
    InferenceRequest,
    InferenceResponse,
    ModelCapabilities,
    ModelRequirements,
    OutputSchema,
)


class CVPRModel(ABC):
    """Universal model adapter interface."""

    @abstractmethod
    def load(self) -> None:
        """Load model resources.

        TODO: Implement loading of model-specific artifacts.
        """

    @abstractmethod
    def infer(self, inputs: InferenceRequest) -> InferenceResponse:
        """Run inference using standardized input payload.

        TODO: Implement model inference entrypoint.
        """

    @abstractmethod
    def validate_inputs(self, inputs: InferenceRequest) -> bool:
        """Validate adapter input payload.

        TODO: Implement strict adapter input validation.
        """

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Return capability metadata for planner compatibility.

        TODO: Implement model capability reporting.
        """

    @abstractmethod
    def get_requirements(self) -> ModelRequirements:
        """Return dependency/runtime requirements.

        TODO: Implement requirement reporting.
        """

    @abstractmethod
    def output_schema(self) -> OutputSchema:
        """Return output schema contract description.

        TODO: Implement schema metadata output.
        """
