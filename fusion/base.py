"""Fusion plugin base contracts.

Defines universal fusion extension points for triangulation, temporal fusion,
volumetric fusion, and neural fusion workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FusionPlugin(ABC):
    """Base interface all fusion plugins must implement."""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize plugin state and dependencies.

        TODO: Implement plugin-specific initialization.
        """

    @abstractmethod
    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Process normalized intermediate outputs.

        TODO: Implement fusion process execution.
        """

    @abstractmethod
    def validate(self, inputs: dict[str, Any]) -> bool:
        """Validate plugin input payload.

        TODO: Implement fusion input validation.
        """

    @abstractmethod
    def output_type(self) -> str:
        """Return semantic output type for downstream routing.

        TODO: Implement output type declaration.
        """
