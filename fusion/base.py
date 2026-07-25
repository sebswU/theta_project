"""Fusion plugin base contracts.

Defines universal fusion extension points for triangulation, temporal fusion,
volumetric fusion, and neural fusion workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas.models import FusionConfiguration, FusionRequest, FusionResponse


class FusionPluginLifecycleError(RuntimeError):
    """Raised when plugin lifecycle methods are used in an invalid order."""


class FusionPlugin(ABC):
    """Base interface all fusion plugins must implement."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Prevent subclasses from bypassing wrapper-level contract checks."""
        super().__init_subclass__(**kwargs)
        overridden = [
            name
            for name in ("initialize", "process", "validate", "output_type")
            if name in cls.__dict__
        ]
        if overridden:
            names = ", ".join(overridden)
            raise TypeError(f"Do not override FusionPlugin wrappers: {names}")

    def __init__(self) -> None:
        self._is_initialized = False
        self._config: FusionConfiguration | None = None

    @property
    def is_initialized(self) -> bool:
        """Return current plugin initialization state."""
        return self._is_initialized

    @property
    def config(self) -> FusionConfiguration:
        """Return active plugin configuration.

        Raises:
            FusionPluginLifecycleError: If called before initialize().
        """
        if self._config is None:
            raise FusionPluginLifecycleError("config accessed before initialize()")
        return self._config

    def initialize(self, config: FusionConfiguration) -> None:
        """Initialize plugin state and dependencies.

        Calling initialize repeatedly is safe and treated as a no-op when already
        initialized.
        """
        if self._is_initialized:
            return
        self._initialize(config)
        self._config = config
        self._is_initialized = True

    def process(self, inputs: FusionRequest) -> FusionResponse:
        """Process normalized intermediate outputs."""
        if not self._is_initialized:
            raise FusionPluginLifecycleError("process() called before initialize()")
        if not self.validate(inputs):
            raise ValueError("Invalid fusion request for plugin")

        output = self._process(inputs)
        if output.request_id != inputs.request_id:
            raise ValueError("Fusion response request_id must match request input")
        return output

    def validate(self, inputs: FusionRequest) -> bool:
        """Validate plugin input payload."""
        if not isinstance(inputs, FusionRequest):
            return False
        if not inputs.request_id.strip():
            return False
        if not inputs.inputs and inputs.scene_graph is None:
            return False
        return self._validate(inputs)

    def output_type(self) -> str:
        """Return semantic output type for downstream routing."""
        value = self._output_type().strip()
        if not value:
            raise ValueError("Fusion plugin output_type must be non-empty")
        return value

    @abstractmethod
    def _initialize(self, config: FusionConfiguration) -> None:
        """Initialize plugin state and dependencies.

        TODO: Implement plugin-specific initialization.
        """

    @abstractmethod
    def _process(self, inputs: FusionRequest) -> FusionResponse:
        """Process normalized intermediate outputs.

        TODO: Implement fusion process execution.
        """

    @abstractmethod
    def _validate(self, inputs: FusionRequest) -> bool:
        """Validate plugin input payload.

        TODO: Implement fusion input validation.
        """

    @abstractmethod
    def _output_type(self) -> str:
        """Return semantic output type for downstream routing.

        TODO: Implement output type declaration.
        """


__all__ = ["FusionPlugin", "FusionPluginLifecycleError"]
