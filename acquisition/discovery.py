"""Discovery provider interfaces for cameras, datasets, ROS, and network streams.

This module now provides a small runtime layer that normalizes provider output
into canonical :class:`schemas.models.SourceDescriptor` objects while keeping
provider-specific discovery behavior isolated in protected hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
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

        Note:
            Discovery does not generate source identifiers. `source_id` values
            are preserved from provider payloads after schema normalization.
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


class SkellyCamDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery adapter for SkellyCam source inventories."""

    def __init__(
        self,
        sources: Iterable[SourceDescriptor | dict[str, Any]]
        | Callable[[], Iterable[SourceDescriptor | dict[str, Any]]],
    ) -> None:
        self._source_loader = sources if callable(sources) else None
        self._static_sources = None if callable(sources) else list(sources)

    def _discover(self) -> Iterable[SourceDescriptor]:
        if self._source_loader is not None:
            discovered = self._source_loader()
        else:
            discovered = self._static_sources or []

        return [_normalize_skellycam_source(item) for item in discovered]


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


def _normalize_skellycam_source(
    value: SourceDescriptor | dict[str, Any],
) -> SourceDescriptor:
    """Convert SkellyCam source data into the canonical source descriptor."""

    if isinstance(value, SourceDescriptor):
        return value
    if not isinstance(value, dict):
        raise TypeError("Unsupported SkellyCam source shape")

    payload = _skellycam_payload_to_source_descriptor(value)

    try:
        return SourceDescriptor.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Invalid SkellyCam source payload") from exc


def _skellycam_payload_to_source_descriptor(value: dict[str, Any]) -> dict[str, Any]:
    """Map common SkellyCam keys into the canonical source descriptor schema."""

    if "source_id" in value and "source_type" in value:
        # Preserve canonical IDs as provided by the upstream source inventory.
        return dict(value)

    if "id" not in value or "type" not in value:
        raise ValueError("SkellyCam source entry is missing required fields")

    metadata = {
        key: item
        for key, item in value.items()
        if key not in {"id", "type", "uri", "source_id", "source_type"}
    }
    payload: dict[str, Any] = {
        # Discovery performs a key mapping only: id -> source_id.
        # IDs are not synthesized, hashed, or otherwise recalculated here.
        "source_id": value["id"],
        "source_type": value["type"],
        "uri": value.get("uri"),
        "metadata": metadata,
    }
    return payload


__all__ = [
    "BaseDiscoveryProvider",
    "CameraDiscoveryProvider",
    "DatasetDiscoveryProvider",
    "DiscoveryLifecycleError",
    "NetworkStreamDiscoveryProvider",
    "ROSDiscoveryProvider",
    "SkellyCamDiscoveryProvider",
]
