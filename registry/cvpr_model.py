"""Abstract base class contract for CVPR/research model adapters.

All future model integrations must implement this interface with a thin
adapter layer only, keeping model-specific code isolated.

TODO:
- Add runtime lifecycle hooks for warm-start and teardown.
- Add async variants for distributed deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CVPRModel(ABC):
    """Universal model adapter interface."""

    @abstractmethod
    def load(self) -> None:
        """Load model resources.

        TODO: Implement loading of model-specific artifacts.
        """

    @abstractmethod
    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference using standardized input payload.

        TODO: Implement model inference entrypoint.
        """

    @abstractmethod
    def validate_inputs(self, inputs: dict[str, Any]) -> bool:
        """Validate adapter input payload.

        TODO: Implement strict adapter input validation.
        """

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return capability metadata for planner compatibility.

        TODO: Implement model capability reporting.
        """

    @abstractmethod
    def get_requirements(self) -> dict[str, Any]:
        """Return dependency/runtime requirements.

        TODO: Implement requirement reporting.
        """

    @abstractmethod
    def output_schema(self) -> dict[str, Any]:
        """Return output schema contract description.

        TODO: Implement schema metadata output.
        """
