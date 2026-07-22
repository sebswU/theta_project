"""Sensor adapter contracts for universal acquisition.

Sensor adapters provide a thin layer between physical/virtual sources and
normalized `schemas.models.Frame` payloads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.models import Frame


class SensorAdapter(ABC):
    """Universal sensor adapter interface."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to underlying source.

        TODO: Implement source connection lifecycle.
        """

    @abstractmethod
    def read(self) -> Frame:
        """Read one frame from source.

        TODO: Implement frame acquisition and normalization.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from source.

        TODO: Implement graceful shutdown behavior.
        """
