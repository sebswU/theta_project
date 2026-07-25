"""Public visualization API for rendering canonical scene graph outputs."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from schemas.models import SceneGraph
from visualization.backends import get_backend


def _coerce_scene_graph(scene_graph: SceneGraph | dict[str, Any]) -> SceneGraph:
	"""Normalize supported scene graph payloads into the canonical schema."""

	if isinstance(scene_graph, SceneGraph):
		return scene_graph
	if isinstance(scene_graph, dict):
		try:
			return SceneGraph.model_validate(scene_graph)
		except ValidationError as exc:
			raise ValueError(
				"Invalid canonical scene graph payload for visualization."
			) from exc

	raise TypeError(
		"scene_graph must be schemas.models.SceneGraph or dict[str, Any]."
	)


def render_scene(
	scene_graph: SceneGraph | dict[str, Any],
	*,
	backend: str = "open3d",
) -> dict[str, Any]:
	"""Render a canonical scene graph through a named visualization backend."""

	canonical_scene = _coerce_scene_graph(scene_graph)
	return get_backend(backend).render(canonical_scene)


__all__ = ["render_scene"]
