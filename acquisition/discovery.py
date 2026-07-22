"""Discovery provider interfaces for cameras, datasets, ROS, and network streams.

This module intentionally defines contracts only; concrete discovery logic
must be implemented by deployment-specific providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDiscoveryProvider(ABC):
    """Base interface for all discovery providers."""

    @abstractmethod
    def discover(self) -> list[dict[str, Any]]:
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
