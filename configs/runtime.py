"""Runtime assembly helpers for declarative configuration files.

The config files under configs/ use JSON syntax, which is a valid YAML subset.
This keeps parsing in the standard library while preserving `.yaml` filenames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from acquisition import BaseDiscoveryProvider
from adapters import SensorAdapter
from orchestration import (
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
    """Resolved runtime objects assembled from declarative config files."""

    sources: list[SourceDescriptor]
    model_registry: ModelRegistry
    plugin_registry: PluginRegistry
    planner: DeterministicPipelinePlanner
    orchestrator: Orchestrator
    plan: PipelinePlan
    graph: WorkflowGraph
    pipeline_name: str
    outputs: list[str]
    source_adapter_classes: dict[str, type[SensorAdapter]]
    source_calibration_refs: dict[str, str]


class StaticDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery provider backed entirely by config-defined source descriptors."""

    def __init__(self, sources: list[SourceDescriptor]) -> None:
        self._sources = list(sources)

    def _discover(self):
        return list(self._sources)


def load_runtime_assembly(
    config_dir: str | Path,
    *,
    pipeline_name: str = "realtime_multicam",
) -> RuntimeAssembly:
    """Load config files and assemble the minimal runtime wiring."""

    base_path = Path(config_dir)
    cameras_config = _load_json_yaml(base_path / "cameras.yaml")
    models_config = _load_json_yaml(base_path / "models.yaml")
    plugins_config = _load_json_yaml(base_path / "plugins.yaml")
    pipelines_config = _load_json_yaml(base_path / "pipelines.yaml")

    sources, source_adapter_classes, source_calibration_refs = _load_sources(
        cameras_config, base_path.parent
    )
    pipelines = pipelines_config["pipelines"]
    pipeline_config = pipelines[pipeline_name]

    selected_sources = _filter_sources(sources, pipeline_config.get("inputs", []))
    selected_source_ids = {source.source_id for source in selected_sources}
    selected_adapter_classes = {
        source_id: adapter_cls
        for source_id, adapter_cls in source_adapter_classes.items()
        if source_id in selected_source_ids
    }
    selected_calibration_refs = {
        source_id: path
        for source_id, path in source_calibration_refs.items()
        if source_id in selected_source_ids
    }

    model_registry, model_requirements = _load_models(models_config)
    plugin_registry, plugin_requirements = _load_plugins(plugins_config)
    _validate_pipeline_references(
        pipeline_config,
        model_requirements=model_requirements,
        plugin_requirements=plugin_requirements,
    )

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
        source_adapter_classes=selected_adapter_classes,
        source_calibration_refs=selected_calibration_refs,
    )


def _load_json_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON/YAML object: {path}")
    return payload


def _load_sources(
    config: dict[str, Any], repo_root: Path
) -> tuple[
    list[SourceDescriptor],
    dict[str, type[SensorAdapter]],
    dict[str, str],
]:
    raw_sources = config.get("sources", [])
    sources: list[SourceDescriptor] = []
    adapter_classes: dict[str, type[SensorAdapter]] = {}
    calibration_refs: dict[str, str] = {}

    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError("Source config entries must be objects")

        source_id = item["id"]
        sensor_type = SensorType(item["type"])
        adapter_path = item.get("adapter")
        if not isinstance(adapter_path, str) or not adapter_path.strip():
            raise ValueError(f"Source '{source_id}' is missing required adapter class path")
        adapter_symbol = _import_symbol(adapter_path)
        if not isinstance(adapter_symbol, type) or not issubclass(adapter_symbol, SensorAdapter):
            raise TypeError(
                f"Configured source adapter for '{source_id}' must inherit SensorAdapter"
            )

        uri = item.get("uri")
        calibration = item.get("calibration")
        if isinstance(calibration, str):
            calibration_path = (repo_root / calibration).resolve()
            if not calibration_path.exists():
                raise ValueError(
                    f"Referenced calibration file does not exist: {calibration}"
                )
            calibration_refs[source_id] = calibration

        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"id", "type", "uri", "adapter"}
        }
        sources.append(
            SourceDescriptor(
                source_id=source_id,
                source_type=sensor_type,
                uri=uri,
                metadata=metadata,
            )
        )
        adapter_classes[source_id] = adapter_symbol

    return sources, adapter_classes, calibration_refs


def _load_models(
    config: dict[str, Any],
) -> tuple[ModelRegistry, dict[str, ModelRequirements]]:
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


def _validate_pipeline_references(
    pipeline_config: dict[str, Any],
    *,
    model_requirements: dict[str, ModelRequirements],
    plugin_requirements: dict[str, dict[str, Any]],
) -> None:
    configured_models = set(model_requirements)
    configured_plugins = set(plugin_requirements)

    requested_models = set(pipeline_config.get("models", []))
    missing_models = sorted(requested_models - configured_models)
    if missing_models:
        raise ValueError(
            "Pipeline references unknown/disabled models: " + ", ".join(missing_models)
        )

    requested_plugins = set(pipeline_config.get("fusion", []))
    missing_plugins = sorted(requested_plugins - configured_plugins)
    if missing_plugins:
        raise ValueError(
            "Pipeline references unknown/disabled fusion plugins: "
            + ", ".join(missing_plugins)
        )


def _filter_sources(
    sources: list[SourceDescriptor], required_types: list[str]
) -> list[SourceDescriptor]:
    required = {SensorType(value) for value in required_types}
    if not required:
        return list(sources)
    filtered = [source for source in sources if source.source_type in required]
    if not filtered:
        raise ValueError("Pipeline input requirements do not match any configured sources")
    return filtered


def _import_symbol(path: str) -> Any:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid import path: {path}")
    module = import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Missing configured symbol: {path}") from exc
