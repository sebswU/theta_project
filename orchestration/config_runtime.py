"""Runtime assembly helpers for declarative configuration files.

This module performs config-driven orchestration wiring and graph creation:
1) Load source/model/plugin/pipeline declarations from ``configs/*.yaml``
    (stored as JSON-compatible YAML).
2) Build typed sources, registries, and requirement maps.
3) Instantiate deterministic planner + orchestrator with pipeline constraints.
4) Execute orchestration to produce a ``PipelinePlan`` and ``WorkflowGraph``.

The result is a single ``RuntimeAssembly`` object containing both configuration
artifacts and the compiled workflow graph for execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from acquisition import BaseDiscoveryProvider
from orchestration.planner import (
    DefaultCapabilityDetector,
    DeterministicPipelinePlanner,
    GraphWorkflowBuilder,
    Orchestrator,
    PlanningConstraints,
)
from registry import ModelRegistry, PluginRegistry
from schemas import ModelRequirements, SensorType, SourceDescriptor, WorkflowGraph
from schemas.models import PipelinePlan


@dataclass(slots=True)
class RuntimeAssembly:
    """Resolved runtime objects assembled from declarative config files.

    Includes discovery inputs, registries, planner/orchestrator instances, and
    the resulting deterministic plan and workflow graph.
    """

    sources: list[SourceDescriptor]
    model_registry: ModelRegistry
    plugin_registry: PluginRegistry
    planner: DeterministicPipelinePlanner
    orchestrator: Orchestrator
    plan: PipelinePlan
    graph: WorkflowGraph
    pipeline_name: str
    outputs: list[str]


class StaticDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery provider backed entirely by config-defined sources.

    Used by runtime assembly to route configured source descriptors through the
    same orchestration path used by dynamic discovery providers.
    """

    def __init__(self, sources: list[SourceDescriptor]) -> None:
        self._sources = list(sources)

    def _discover(self):
        """Return configured sources as-is for deterministic orchestration."""
        return list(self._sources)


def load_runtime_assembly(
    config_dir: str | Path,
    *,
    pipeline_name: str = "realtime_multicam",
) -> RuntimeAssembly:
    """Load configs and assemble runtime wiring plus compiled workflow graph.

    The named pipeline contributes candidate constraints:
    - ``models`` limits selectable model adapters.
    - ``fusion`` limits selectable fusion plugins.
    - ``inputs`` filters which configured sources are considered.

    The orchestrator then builds ``(plan, graph)`` by running capability
    detection, deterministic planning, and workflow graph compilation.
    """

    base_path = Path(config_dir)
    cameras_config = _load_json_yaml(base_path / "cameras.yaml")
    models_config = _load_json_yaml(base_path / "models.yaml")
    plugins_config = _load_json_yaml(base_path / "plugins.yaml")
    pipelines_config = _load_json_yaml(base_path / "pipelines.yaml")

    sources = _load_sources(cameras_config)
    pipelines = pipelines_config["pipelines"]
    pipeline_config = pipelines[pipeline_name]

    selected_sources = _filter_sources(sources, pipeline_config.get("inputs", []))

    model_registry, model_requirements = _load_models(models_config)
    plugin_registry, plugin_requirements = _load_plugins(plugins_config)

    planner = DeterministicPipelinePlanner(
        model_requirements=model_requirements,
        plugin_requirements=plugin_requirements,
        constraints=PlanningConstraints(
            allowed_models=set(pipeline_config.get("models", [])),
            allowed_plugins=set(pipeline_config.get("fusion", [])),
        ),
    )
    orchestrator = Orchestrator(
        capability_detector=DefaultCapabilityDetector(),
        planner=planner,
        workflow_builder=GraphWorkflowBuilder(),
    )

    discovery_provider = StaticDiscoveryProvider(selected_sources)
    plan, graph = orchestrator.create_workflow(discovery_provider.discover())

    _validate_referenced_files(selected_sources, base_path.parent)

    return RuntimeAssembly(
        sources=selected_sources,
        model_registry=model_registry,
        plugin_registry=plugin_registry,
        planner=planner,
        orchestrator=orchestrator,
        plan=plan,
        graph=graph,
        pipeline_name=pipeline_name,
        outputs=list(pipeline_config.get("outputs", [])),
    )


def _load_json_yaml(path: Path) -> dict[str, Any]:
    """Load one JSON-compatible YAML file into a mapping object."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON/YAML object: {path}")
    return payload


def _load_sources(config: dict[str, Any]) -> list[SourceDescriptor]:
    """Convert camera/source config objects into typed source descriptors."""
    raw_sources = config.get("sources", [])
    sources: list[SourceDescriptor] = []

    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("Source config entries must be objects")

        # Config loading maps external `id` directly to canonical `source_id`.
        # Runtime assembly does not generate new source identifiers.
        source_id = item["id"]
        sensor_type = SensorType(item["type"])
        uri = item.get("uri")
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"id", "type", "uri"}
        }
        sources.append(
            SourceDescriptor(
                source_id=source_id,
                source_type=sensor_type,
                uri=uri,
                metadata=metadata,
            )
        )
    return sources


def _validate_referenced_files(sources: list[SourceDescriptor], repo_root: Path) -> None:
    """Validate that source metadata file references exist on disk.

    Currently validates optional calibration paths when provided as strings.
    """
    for source in sources:
        calibration = source.metadata.get("calibration")
        if isinstance(calibration, str):
            calibration_path = (repo_root / calibration).resolve()
            if not calibration_path.exists():
                raise ValueError(f"Referenced calibration file does not exist: {calibration}")


def _load_models(
    config: dict[str, Any],
) -> tuple[ModelRegistry, dict[str, ModelRequirements]]:
    """Load enabled models into the registry and requirement map.

    The requirement map feeds planner capability matching; registry entries
    enable later runtime lookup/instantiation by model name.
    """
    registry = ModelRegistry()
    requirements: dict[str, ModelRequirements] = {}

    for entry in config.get("models", []):
        if not entry.get("enabled", False):
            continue
        model_name = entry["name"]
        model_cls = _import_symbol(entry["adapter"])
        registry.register(model_name, model_cls)
        requirements[model_name] = ModelRequirements(
            model_name=model_name,
            required_sources=[SensorType(value) for value in entry.get("required_sources", [])],
            resources={
                "device": entry.get("device"),
                "precision": entry.get("precision"),
                "dependencies": entry.get("dependencies", []),
            },
            metadata={"adapter": entry["adapter"]},
        )
    return registry, requirements


def _load_plugins(
    config: dict[str, Any],
) -> tuple[PluginRegistry, dict[str, dict[str, Any]]]:
    """Load enabled plugins into the registry and requirement map.

    Plugin requirements are consumed by planner subset checks against available
    source sensor types.
    """
    registry = PluginRegistry()
    requirements: dict[str, dict[str, Any]] = {}

    for entry in config.get("plugins", []):
        if not entry.get("enabled", False):
            continue
        plugin_name = entry["name"]
        plugin_cls = _import_symbol(entry["class"])
        registry.register(plugin_name, plugin_cls)
        requirements[plugin_name] = {
            "required_sources": entry.get("required_sources", []),
            "stage": entry.get("stage"),
            "class": entry["class"],
        }
    return registry, requirements


def _filter_sources(
    sources: list[SourceDescriptor], required_types: list[str]
) -> list[SourceDescriptor]:
    """Filter configured sources to those required by the selected pipeline.

    Returns all sources when no input filter is declared.
    """
    required = {SensorType(value) for value in required_types}
    if not required:
        return list(sources)
    filtered = [source for source in sources if source.source_type in required]
    if not filtered:
        raise ValueError("Pipeline input requirements do not match any configured sources")
    return filtered


def _import_symbol(path: str) -> Any:
    """Import and return a symbol from a ``module.attribute`` string path."""
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid import path: {path}")
    module = import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Missing configured symbol: {path}") from exc
