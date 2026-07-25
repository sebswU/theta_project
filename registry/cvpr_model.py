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


class CVPRModelLifecycleError(RuntimeError):
    """Raised when model lifecycle methods are used in an invalid order."""


class CVPRModel(ABC):
    """Universal model adapter interface with registry-facing validation wrappers."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Prevent subclasses from bypassing wrapper-level contract checks."""
        super().__init_subclass__(**kwargs)
        overridden = [
            name
            for name in (
                "load",
                "infer",
                "validate_inputs",
                "get_capabilities",
                "get_requirements",
                "output_schema",
            )
            if name in cls.__dict__
        ]
        if overridden:
            names = ", ".join(overridden)
            raise TypeError(f"Do not override CVPRModel wrappers: {names}")

    def __init__(self) -> None:
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return current model load state."""
        return self._is_loaded

    def load(self) -> None:
        """Load model resources.

        Calling load repeatedly is safe and treated as a no-op when loaded.
        """
        if self._is_loaded:
            return
        self._load()
        self._is_loaded = True

    def infer(self, inputs: InferenceRequest) -> InferenceResponse:
        """Run inference using standardized input payload."""
        if not self._is_loaded:
            raise CVPRModelLifecycleError("infer() called before load()")
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inference request for model adapter")
        return self._infer(inputs)

    def validate_inputs(self, inputs: InferenceRequest) -> bool:
        """Validate adapter input payload."""
        if not isinstance(inputs, InferenceRequest):
            return False
        if not inputs.request_id.strip():
            return False
        if not inputs.frames:
            return False
        return self._validate_inputs(inputs)

    def get_capabilities(self) -> ModelCapabilities:
        """Return capability metadata for planner compatibility."""
        capabilities = self._get_capabilities()
        if capabilities.model_name != self.model_name:
            raise ValueError(
                "Capability metadata model_name must match adapter model_name"
            )
        return capabilities

    def get_requirements(self) -> ModelRequirements:
        """Return dependency/runtime requirements."""
        requirements = self._get_requirements()
        if requirements.model_name != self.model_name:
            raise ValueError(
                "Requirement metadata model_name must match adapter model_name"
            )
        return requirements

    def output_schema(self) -> OutputSchema:
        """Return output schema contract description."""
        return self._output_schema()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable adapter/model identifier used by registry metadata."""

    @abstractmethod
    def _load(self) -> None:
        """Load model resources.

        TODO: Implement loading of model-specific artifacts.
        """

    @abstractmethod
    def _infer(self, inputs: InferenceRequest) -> InferenceResponse:
        """Run inference using standardized input payload.

        TODO: Implement model inference entrypoint.
        """

    @abstractmethod
    def _validate_inputs(self, inputs: InferenceRequest) -> bool:
        """Validate adapter input payload.

        TODO: Implement strict adapter input validation.
        """

    @abstractmethod
    def _get_capabilities(self) -> ModelCapabilities:
        """Return capability metadata for planner compatibility.

        TODO: Implement model capability reporting.
        """

    @abstractmethod
    def _get_requirements(self) -> ModelRequirements:
        """Return dependency/runtime requirements.

        TODO: Implement requirement reporting.
        """

    @abstractmethod
    def _output_schema(self) -> OutputSchema:
        """Return output schema contract description.

        TODO: Implement schema metadata output.
        """


__all__ = ["CVPRModel", "CVPRModelLifecycleError"]
