"""Capability detector contracts.

Capability detectors translate discovered metadata into typed capability
profiles used for planning and model/fusion selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from schemas.capabilities import SensorCapabilityProfile


class CapabilityDetector(ABC):
    """Converts source descriptors into capability profiles."""

    @abstractmethod
    def detect(self, source_descriptor: dict[str, Any]) -> SensorCapabilityProfile:
        """Build a typed capability profile.

        TODO: Implement detection and confidence annotation.
        """
