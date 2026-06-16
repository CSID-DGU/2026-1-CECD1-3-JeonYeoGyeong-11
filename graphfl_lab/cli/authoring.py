"""Scaffold and compose helpers for Graph-FL component authoring."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KINDS = {"source", "builder", "aggregation"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def poster_session_root() -> Path:
    return repository_root() / "tmp" / "poster_sessions"


def _safe_identifier(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(
            f"{label} must be a Python identifier: {value!r}"
        )
    return text


def _safe_workspace(workspace: str | Path | None, session_id: str | None) -> Path:
    root = poster_session_root().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if workspace:
        target = Path(workspace).expanduser()
        if not target.is_absolute():
            target = repository_root() / target
        target = target.resolve()
    else:
        session = _safe_identifier(
            session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S_%f"),
            label="session id",
        )
        target = (root / session).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"workspace must stay under {root}: {target}"
        ) from exc
    return target


def _source_template(name: str) -> str:
    return f'''"""Poster-ready graph source: {name}."""

from __future__ import annotations

import numpy as np

from graphfl_lab.extensions import (
    GraphSourceContext,
    GraphSourceResult,
    flatten_client_arrays,
    register_graph_source,
)

FEATURE_COUNT = 3


@register_graph_source("{name}")
def build_{name}(ctx: GraphSourceContext) -> GraphSourceResult:
    vectors = []
    for client_arrays in ctx.local_updates:
        flat = flatten_client_arrays(client_arrays).astype(np.float64, copy=False)
        # AUTHORING TODO: choose two or three update statistics.
        features = [float(np.mean(flat)), float(np.std(flat))]
        if FEATURE_COUNT == 3:
            features.append(float(np.mean(flat > 0.0)))
        vectors.append(np.asarray(features, dtype=np.float64))
    return GraphSourceResult(
        vectors=vectors,
        source_used="{name}",
        metadata={{"features": FEATURE_COUNT}},
    )
'''


def _builder_template(name: str) -> str:
    return f'''"""Poster-ready graph builder: {name}."""

from __future__ import annotations

import numpy as np

from graphfl_lab.extensions import (
    GraphBuildContext,
    GraphBuildResult,
    register_graph_builder,
)


@register_graph_builder("{name}")
def build_{name}(ctx: GraphBuildContext) -> GraphBuildResult:
    z = np.asarray(ctx.z_mat, dtype=np.float64)
    safe = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    similarity = np.maximum(safe @ safe.T, 0.0)
    np.fill_diagonal(similarity, 0.0)
    k = min(max(int(ctx.knn_k), 1), max(z.shape[0] - 1, 1))
    # AUTHORING TODO: retain an edge only when both clients select each other.
    neighbors = np.zeros_like(similarity, dtype=bool)
    neighbors[np.arange(z.shape[0])[:, None], np.argsort(similarity, axis=1)[:, -k:]] = True
    mutual = neighbors & neighbors.T
    adjacency = np.where(mutual, similarity, 0.0)
    return GraphBuildResult(
        adjacency=adjacency,
        metadata={{"graph_kind": "plugin:{name}", "k": k}},
    )
'''


def _aggregation_template(name: str) -> str:
    return f'''"""Poster-ready aggregation target: {name}."""

from __future__ import annotations

from graphfl_lab.extensions import (
    AggregationTargetContext,
    AggregationTargetResult,
    graph_filter_client_arrays,
    mix_client_arrays,
    register_aggregation_target,
)


@register_aggregation_target("{name}")
def aggregate_{name}(ctx: AggregationTargetContext) -> AggregationTargetResult:
    beta = float(ctx.config.parameters.get("beta", 0.5))
    filtered, filter_meta = graph_filter_client_arrays(
        ctx.local_updates,
        ctx.l_mat,
        ctx.config.filter_strength,
        target_name="{name}",
    )
    # AUTHORING TODO: mix FedAvg updates with graph-filtered updates.
    mixed = mix_client_arrays(ctx.local_updates, filtered, beta)
    return AggregationTargetResult(
        post_local_updates=mixed,
        target_used="{name}",
        metadata={{"beta": beta, **filter_meta}},
    )
'''


def _contract_readme(kind: str, name: str) -> str:
    details = {
        "source": (
            "`GraphSourceContext`에서 client별 local arrays를 읽고 "
            "`GraphSourceResult`를 반환합니다.\n\n"
            "- vector 수는 client 수와 같아야 합니다.\n"
            "- 모든 vector 길이는 같고 finite여야 합니다.\n"
            "- client 순서와 입력 array를 변경하지 않습니다."
        ),
        "builder": (
            "`GraphBuildContext.z_mat`에서 `(N, N)` adjacency를 만듭니다.\n\n"
            "- finite, non-negative, symmetric, zero diagonal을 만족합니다.\n"
            "- `knn_k`와 plugin metadata를 기록합니다."
        ),
        "aggregation": (
            "`AggregationTargetContext`에서 client별 post-update를 만듭니다.\n\n"
            "- client/layer shape를 그대로 유지합니다.\n"
            "- 실제 aggregation과 diagnostics가 같은 post-update를 사용합니다.\n"
            "- parameter와 target metadata를 기록합니다."
        ),
    }[kind]
    return f"""# {kind}: `{name}`

## Contract

{details}

## Authoring Flow

1. `AUTHORING TODO` 블록을 수정합니다.
2. `graphfl component validate <plugin-path> --component {kind}:{name}`를 실행합니다.
3. Registry, Shape, Finite, Metadata, Trace/Artifact가 모두 PASS인지 확인합니다.
"""


def _contract_test(kind: str, name: str, plugin_path: Path) -> str:
    return f'''import json
import subprocess
import sys
import unittest


class GeneratedContractTest(unittest.TestCase):
    def test_component_contract(self):
        command = [
            sys.executable,
            "-m",
            "graphfl_lab.extensions.validation_worker",
            "--plugin",
            {str(plugin_path)!r},
            "--component",
            "{kind}:{name}",
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"], payload)


if __name__ == "__main__":
    unittest.main()
'''


def scaffold_component(
    *,
    kind: str,
    name: str,
    workspace: str | Path | None = None,
    session_id: str | None = None,
    plugin_name: str = "poster_plugin",
) -> Mapping[str, str]:
    kind = str(kind).strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"component kind must be one of {sorted(_KINDS)}")
    name = _safe_identifier(name, label="component name")
    plugin_name = _safe_identifier(plugin_name, label="plugin name")
    session_dir = _safe_workspace(workspace, session_id)
    plugin_dir = session_dir / plugin_name
    kind_dir = plugin_dir / {
        "source": "sources",
        "builder": "builders",
        "aggregation": "aggregations",
    }[kind]
    component_path = kind_dir / f"{name}.py"
    readme_path = kind_dir / f"{name}.README.md"
    test_path = session_dir / "tests" / f"test_{kind}_{name}_contract.py"
    for path in (component_path, readme_path, test_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    kind_dir.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    package_init = plugin_dir / "__init__.py"
    kind_init = kind_dir / "__init__.py"
    if not package_init.exists():
        package_init.write_text('"""Generated Graph-FL plugin package."""\n', encoding="utf-8")
    if not kind_init.exists():
        kind_init.write_text("", encoding="utf-8")

    template = {
        "source": _source_template,
        "builder": _builder_template,
        "aggregation": _aggregation_template,
    }[kind](name)
    component_path.write_text(template, encoding="utf-8")
    readme_path.write_text(_contract_readme(kind, name), encoding="utf-8")
    test_path.write_text(
        _contract_test(kind, name, plugin_dir.resolve()),
        encoding="utf-8",
    )

    import_line = f"from .{kind_dir.name} import {name} as _{kind}_{name}\n"
    existing = package_init.read_text(encoding="utf-8")
    if import_line not in existing:
        package_init.write_text(existing + import_line, encoding="utf-8")

    return {
        "session_dir": str(session_dir),
        "plugin_path": str(plugin_dir),
        "component_path": str(component_path),
        "readme_path": str(readme_path),
        "test_path": str(test_path),
        "component": f"{kind}:{name}",
    }


def compose_design(
    *,
    plugin: str | Path,
    name: str,
    source: str,
    builder: str,
    aggregation: str,
    knn_k: int = 2,
    aggregation_params: Mapping[str, Any] | None = None,
) -> Mapping[str, str]:
    name = _safe_identifier(name, label="design name")
    source = _safe_identifier(source, label="source name")
    builder = _safe_identifier(builder, label="builder name")
    aggregation = _safe_identifier(aggregation, label="aggregation name")
    plugin_dir = Path(plugin).expanduser().resolve()
    if not plugin_dir.is_dir() or not (plugin_dir / "__init__.py").is_file():
        raise ValueError(f"plugin must be a Python package: {plugin_dir}")
    design_path = plugin_dir / "designs.py"
    if design_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {design_path}")
    params = dict(aggregation_params or {})
    session_dir = plugin_dir.parent
    config_path = session_dir / f"{name}.json"
    if config_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {config_path}")
    design_code = f'''"""Generated GraphFLDesign composition."""

from graphfl_lab.extensions import ComponentSpec, GraphFLDesign, register_design


register_design(
    GraphFLDesign(
        name={name!r},
        client_state=ComponentSpec(
            kind="ClientStateExtractor",
            name={source!r},
            params={{"graph_source": {source!r}}},
            output_kind="client_state",
        ),
        relation=ComponentSpec(
            kind="RelationEstimator",
            name="cosine_similarity",
            output_kind="relation",
        ),
        topology=ComponentSpec(
            kind="TopologyOperator",
            name={builder!r},
            params={{"graph_mode": {builder!r}, "knn_k": {int(knn_k)}}},
            input_kind=("client_state",),
            output_kind="topology",
        ),
        aggregation=ComponentSpec(
            kind="AggregationOperator",
            name={aggregation!r},
            params={{
                "aggregation_target": {aggregation!r},
                "aggregation_params": {params!r},
            }},
            input_kind=("topology", "local_update"),
            output_kind="global_weights",
        ),
        tags=("generated", "poster-authoring"),
        description="Generated by graphfl design compose",
    ),
    override=True,
)
'''
    design_path.write_text(design_code, encoding="utf-8")
    package_init = plugin_dir / "__init__.py"
    existing = package_init.read_text(encoding="utf-8")
    import_line = "from . import designs as _designs\n"
    if import_line not in existing:
        package_init.write_text(existing + import_line, encoding="utf-8")
    config_payload = {
        "schema_version": 2,
        "args": {
            "method": "ours",
            "graph_plugin": str(plugin_dir),
            "graph_preset": name,
            "graph_source": source,
            "graph_mode": builder,
            "aggregation_target": aggregation,
            "aggregation_params": params,
            "knn_k": int(knn_k),
        },
        "authoring": {
            "component_source": source,
            "component_builder": builder,
            "component_aggregation": aggregation,
            "plugin": str(plugin_dir),
            "design": name,
        },
    }
    config_path.write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "plugin_path": str(plugin_dir),
        "design_path": str(design_path),
        "design_name": name,
        "config_path": str(config_path),
        "run_command": (
            f"graphfl run single --track vision --config {config_path} --dry-run"
        ),
    }


def validate_component(
    *,
    plugin: str | Path,
    component: str | None = None,
    timeout: float = 30.0,
) -> Mapping[str, Any]:
    command = [
        sys.executable,
        "-m",
        "graphfl_lab.extensions.validation_worker",
        "--plugin",
        str(Path(plugin).expanduser()),
    ]
    if component:
        command.extend(["--component", component])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout),
        cwd=repository_root(),
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"component validation failed: {detail}") from None
    if completed.returncode not in {0, 1}:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"component validation failed: {detail}")
    return payload


__all__ = [
    "compose_design",
    "poster_session_root",
    "repository_root",
    "scaffold_component",
    "validate_component",
]
