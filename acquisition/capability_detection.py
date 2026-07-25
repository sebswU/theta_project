"""Capability detector contracts.

Capability detectors translate discovered metadata into typed capability
profiles used for planning and model/fusion selection. The public wrapper in
this module now normalizes detector output into canonical
``SensorCapabilityProfile`` objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from schemas.capabilities import SensorCapabilityProfile
from schemas.models import SourceDescriptor


class CapabilityDetectionError(RuntimeError):
    """Raised when capability detection hooks are bypassed or misused."""


class CapabilityDetector(ABC):
    """Converts source descriptors into capability profiles."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Prevent subclasses from bypassing the public detection wrapper."""
        super().__init_subclass__(**kwargs)
        if "detect" in cls.__dict__:
            raise TypeError("Do not override detection wrapper: detect")

    def detect(self, source_descriptor: SourceDescriptor) -> SensorCapabilityProfile:
        """Build a typed capability profile.

        The public wrapper preserves the source identity/type and lets the hook
        provide additional capability flags or metadata.
        """
        if not isinstance(source_descriptor, SourceDescriptor):
            raise TypeError("source_descriptor must be SourceDescriptor")

        detected = self._detect(source_descriptor)
        payload: dict[str, Any]
        if isinstance(detected, SensorCapabilityProfile):
            payload = detected.model_dump()
        elif isinstance(detected, dict):
            payload = dict(detected)
        else:
            raise TypeError("Unsupported capability profile shape")

        payload.setdefault("source_id", source_descriptor.source_id)
        payload.setdefault("sensor_type", source_descriptor.source_type)
        allowed_fields = set(SensorCapabilityProfile.model_fields)
        payload = {key: value for key, value in payload.items() if key in allowed_fields}

        try:
            profile = SensorCapabilityProfile.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("Invalid capability profile payload") from exc

        if profile.source_id != source_descriptor.source_id:
            raise ValueError("Capability profile source_id must match descriptor")
        if profile.sensor_type != source_descriptor.source_type:
            raise ValueError("Capability profile sensor_type must match descriptor")
        return profile

    @abstractmethod
    def _detect(
        self, source_descriptor: SourceDescriptor
    ) -> SensorCapabilityProfile | dict[str, Any]:
        """Build a typed capability profile.

        TODO: Implement detection and confidence annotation.
        """


__all__ = ["CapabilityDetector", "CapabilityDetectionError"]
