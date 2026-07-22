"""Strongly typed capability definitions for sensors and inputs.

These data models normalize capability metadata so orchestration can
perform deterministic planning across heterogeneous providers.

TODO:
- Add richer synchronization and timestamp quality metrics.
- Add bandwidth, latency, and calibration quality scoring fields.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SensorType(StrEnum):
    """Supported sensor/source families."""

    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    THERMAL_CAMERA = "thermal_camera"
    IMU = "imu"
    LIDAR = "lidar"
    DATASET = "dataset"
    NETWORK_STREAM = "network_stream"


class SensorCapabilityProfile(BaseModel):
    """Normalized capability object used by planners and matchers."""

    source_id: str = Field(..., description="Unique source identifier")
    sensor_type: SensorType
    supports_rgb: bool = False
    supports_depth: bool = False
    supports_thermal: bool = False
    supports_imu: bool = False
    supports_synchronization: bool = False
    supports_calibration: bool = False
    supports_multiview: bool = False
