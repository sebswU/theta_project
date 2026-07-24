"""Fusion plugin base contracts.

Defines universal fusion extension points for triangulation, temporal fusion,
volumetric fusion, and neural fusion workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.models import FusionConfiguration, FusionRequest, FusionResponse


class FusionPlugin(ABC):
    """Base interface all fusion plugins must implement."""

    @abstractmethod
    def initialize(self, config: FusionConfiguration) -> None:
        """Initialize plugin state and dependencies.

        TODO: Implement plugin-specific initialization.
        """

    @abstractmethod
    def process(self, inputs: FusionRequest) -> FusionResponse:
        """Process normalized intermediate outputs.

        TODO: Implement fusion process execution.
        """

    @abstractmethod
    def validate(self, inputs: FusionRequest) -> bool:
        """Validate plugin input payload.

        TODO: Implement fusion input validation.
        """

    @abstractmethod
    def output_type(self) -> str:
        """Return semantic output type for downstream routing.

        TODO: Implement output type declaration.
        """
