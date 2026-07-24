"""Sensor adapter contracts for universal acquisition.

Sensor adapters provide a thin layer between physical/virtual sources and
normalized `schemas.models.Frame` payloads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.models import Frame, SourceDescriptor


class SensorAdapterLifecycleError(RuntimeError):
    """Raised when adapter lifecycle methods are used in an invalid order."""


class SensorAdapter(ABC):
    """Universal sensor adapter interface with enforced lifecycle ordering."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Prevent subclasses from bypassing lifecycle checks by overriding wrappers."""
        super().__init_subclass__(**kwargs)
        lifecycle_methods = ("connect", "read", "disconnect")
        overridden = [name for name in lifecycle_methods if name in cls.__dict__]
        if overridden:
            names = ", ".join(overridden)
            raise TypeError(f"Do not override lifecycle wrappers: {names}")

    def __init__(self, source: SourceDescriptor) -> None:
        self._source = source
        self._is_connected = False

    @property
    def source(self) -> SourceDescriptor:
        """Return the typed source descriptor this adapter is bound to."""
        return self._source

    @property
    def is_connected(self) -> bool:
        """Return current adapter connection state."""
        return self._is_connected

    def connect(self) -> None:
        """Connect to underlying source.

        Calling connect repeatedly is safe and treated as a no-op when already connected.
        """
        if self._is_connected:
            return
        self._connect()
        self._is_connected = True

    def read(self) -> Frame:
        """Read one frame from source.

        Raises:
            SensorAdapterLifecycleError: If called before connect().
        """
        if not self._is_connected:
            raise SensorAdapterLifecycleError("read() called before connect()")
        return self._read()

    def disconnect(self) -> None:
        """Disconnect from source.

        Disconnect is idempotent and safe to call repeatedly.
        """
        if not self._is_connected:
            return
        try:
            self._disconnect()
        finally:
            self._is_connected = False

    @abstractmethod
    def _connect(self) -> None:
        """Connect to underlying source implementation.

        TODO: Implement source connection lifecycle.
        """

    @abstractmethod
    def _read(self) -> Frame:
        """Read one frame from source implementation.

        TODO: Implement frame acquisition and normalization.
        """

    @abstractmethod
    def _disconnect(self) -> None:
        """Disconnect from source implementation.

        TODO: Implement graceful shutdown behavior.
        """


__all__ = ["SensorAdapter", "SensorAdapterLifecycleError"]
