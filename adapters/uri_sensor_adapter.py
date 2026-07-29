"""Concrete sensor adapter that connects using SourceDescriptor.uri.

This scaffold adapter demonstrates how URI fields from discovery/config are
consumed by acquisition runtime logic. It validates URI shape on connect and
emits deterministic frame metadata that includes the bound endpoint.
"""

from __future__ import annotations

from pathlib import Path
from time import time_ns
from urllib.parse import urlparse

from adapters.sensor_adapter import SensorAdapter, SensorAdapterLifecycleError
from schemas.models import Frame


class URISensorAdapter(SensorAdapter):
    """Sensor adapter that uses ``source.uri`` as its connection endpoint."""

    _NETWORK_SCHEMES = {"rtsp", "http", "https", "udp", "tcp"}

    def __init__(self, source):
        super().__init__(source)
        self._connected_uri: str | None = None
        self._connection_kind: str | None = None
        self._read_index = 0

    def _connect(self) -> None:
        uri = (self.source.uri or "").strip()
        if not uri:
            raise ValueError("Source URI is required to connect")

        parsed = urlparse(uri)
        if parsed.scheme:
            scheme = parsed.scheme.lower()
            if scheme == "file":
                if not parsed.path:
                    raise ValueError("File URI must include a path")
                self._connection_kind = "file"
            elif scheme in self._NETWORK_SCHEMES:
                if not parsed.netloc:
                    raise ValueError(f"URI scheme '{scheme}' must include network location")
                self._connection_kind = "network"
            else:
                raise ValueError(f"Unsupported URI scheme: {scheme}")
        else:
            # Treat plain strings without a scheme as local file or device paths.
            normalized = str(Path(uri).expanduser())
            if not normalized.strip():
                raise ValueError("Local URI/path is empty")
            uri = normalized
            self._connection_kind = "path"

        self._connected_uri = uri
        self._read_index = 0

    def _read(self) -> Frame:
        if self._connected_uri is None or self._connection_kind is None:
            raise SensorAdapterLifecycleError("Adapter is not connected to a URI")

        self._read_index += 1
        return Frame(
            frame_id=f"{self.source.source_id}-{self._read_index:06d}",
            timestamp_ns=time_ns(),
            source_id=self.source.source_id,
            payload={
                "uri": self._connected_uri,
                "connection_kind": self._connection_kind,
                "sequence": self._read_index,
            },
        )

    def _disconnect(self) -> None:
        self._connected_uri = None
        self._connection_kind = None


__all__ = ["URISensorAdapter"]
