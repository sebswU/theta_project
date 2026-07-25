# Developer Guide

## Principles

- Keep model integrations thin and isolated.
- Favor typed schemas over ad-hoc dicts.
- Route all planning decisions through orchestration interfaces.

## Extension Workflow

1. Add adapter class inheriting `CVPRModel`.
2. Register adapter in `ModelRegistry` and `configs/models.yaml`.
3. Add schema mappings and update tests.

## Visualization Entry Point

The public visualization API is render_scene(scene_graph, backend="open3d") in
visualization/__init__.py.

Contract:
- scene_graph accepts either the canonical schemas.models.SceneGraph instance or
	a dict payload matching the same schema.
- backend selects the renderer implementation.
- return value is a deterministic summary dict for scaffold-level integrations
	while runtime-specific rendering is still under implementation.

Supported backends:
- open3d
- blender
- web_dashboard

Failure behavior:
- Invalid scene_graph payloads raise ValueError with a clear canonical payload
	validation message.
- Unsupported backend values raise ValueError listing supported names.

## TODO

- TODO: Add coding standards and code owners.
