"""Discovery provider interfaces for cameras, datasets, ROS, and network streams.

This module now provides a small runtime layer that normalizes provider output
into canonical :class:`schemas.models.SourceDescriptor` objects while keeping
provider-specific discovery behavior isolated in protected hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from schemas.models import SourceDescriptor


class DiscoveryLifecycleError(RuntimeError):
    """Raised when discovery hooks are bypassed or misused."""


class BaseDiscoveryProvider(ABC):
    """Base interface for all discovery providers."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Prevent subclasses from bypassing the public discovery wrapper."""
        super().__init_subclass__(**kwargs)
        if "discover" in cls.__dict__:
            raise TypeError("Do not override discovery wrapper: discover")

    def discover(self) -> list[SourceDescriptor]:
        """Discover source endpoints and metadata.

        The public wrapper normalizes and sorts all discovered descriptors so
        downstream consumers receive a stable source inventory.
        """
        discovered = self._discover()
        normalized = [_coerce_source_descriptor(item) for item in discovered]
        return sorted(normalized, key=lambda item: item.source_id)

    @abstractmethod
    def _discover(self) -> Iterable[SourceDescriptor | dict[str, Any]]:
        """Discover source endpoints and metadata.

        TODO: Implement provider-specific discovery behavior.
        """


class CameraDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery contract for USB/depth/RGB camera sources."""


class DatasetDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery contract for local/remote datasets."""


class ROSDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery contract for ROS2 topics and message streams."""


class NetworkStreamDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery contract for RTSP/WebRTC/network camera sources."""


def _coerce_source_descriptor(
    value: SourceDescriptor | dict[str, Any],
) -> SourceDescriptor:
    """Convert provider output into the canonical source descriptor type."""
    if isinstance(value, SourceDescriptor):
        return value
    if isinstance(value, dict):
        try:
            return SourceDescriptor.model_validate(value)
        except ValidationError as exc:
            raise ValueError("Invalid source descriptor payload") from exc
    raise TypeError("Unsupported source descriptor shape")


__all__ = [
    "BaseDiscoveryProvider",
    "CameraDiscoveryProvider",
    "DatasetDiscoveryProvider",
    "DiscoveryLifecycleError",
    "NetworkStreamDiscoveryProvider",
    "ROSDiscoveryProvider",
]
