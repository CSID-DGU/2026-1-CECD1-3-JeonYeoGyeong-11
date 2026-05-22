# Naming And Compatibility

Gate 6 hard cleanup and post-Gate-6 alias trim are **complete** on `main`
(2026-05-22). Removal history and tombstones:
[`docs/removed-materials.md`](../removed-materials.md). The maintenance
execution log is **closed**:
[`docs/maintenance/cleanup-status.md`](../maintenance/cleanup-status.md).

## Canonical Names

Use canonical names in new code, docs, configs, and reports.

| Concept | Canonical |
|---|---|
| Package root | `graphfl_lab` |
| Vision experiment implementation | `graphfl_lab/experiments/vision/` |
| Vision suite grammar/reporting | `graphfl_lab/experiments/suites/vision/` |
| Vision CLI parser modules | `graphfl_lab/cli/vision_*.py` |
| Vision launchers | `run_vision_*.py` |
| Framework docs | `docs/framework/` |
| Research notes | `docs/research/` |
| Archive | `docs/archive/` |
| Graph-FL strategy runtime | `graphfl_lab/strategies/graphfl/` |
| Graph-FL runtime class | `GraphFLDiagnosticStrategy` |
| Graph vector helper | `graph_vectors_for_graphfl` (`graphfl_lab/graph/sources/graphfl.py`) |
| Aggregation targets | `graph_filtered_update`, `graph_filtered_ema_update`, `graph_filtered_weight` |
| Filter strength key | `graph_filter_strength` (CLI: `--graph-filter-strength`) |
| Suite token family | `ours_graph_filtered_*` |
| Filter-only suffix | `_graph_filter_only` |
| Result filenames | `result_vision_*.json` |
| Suite artifact filenames | `vision_suite_summary.*`, `vision_suite_rows.json` |
| Validation scripts | `scripts/checks/` |
| Smoke scripts | `scripts/smoke/` |
| Historical analysis scripts | `scripts/archive/legacy-analysis/` |

## Removed Surfaces (Do Not Reintroduce)

These names are **not** supported in active code paths. See
`docs/removed-materials.md` for batch notes.

| Removed | Replacement |
|---|---|
| `spectral_fl` package shim | `graphfl_lab` |
| `run_general_*`, `graphfl_lab/general_*`, `cli/general_*` | `run_vision_*`, `vision_*` modules |
| `result_general_*`, `general_suite_*` readers/writers | `result_vision_*`, `vision_suite_*` |
| `graphfl_lab/strategies/spectral/` facades | `graphfl_lab/strategies/graphfl/` |
| CLI `spectral_filtered_*` choices; `--spectral-filter-strength` | `graph_filtered_*`; `--graph-filter-strength` |
| Suite launch tokens `ours_spectral_filtered_*` | `ours_graph_filtered_*` |
| `graph_vectors_for_spectral`, `graph/sources/spectral.py` | `graph_vectors_for_graphfl` |

## Remaining Read-Only Compatibility

These exist only to load **historical** configs, aggregation target strings, or
result tags. New runs should not depend on them.

| Compatibility surface | Mechanism | Notes |
|---|---|---|
| `configs/general/...` | path alias in `config_io` | resolves to `configs/vision/...` |
| `spectral_filter_strength` | JSON key alias in `config_io` | maps to `graph_filter_strength`; not written in new result meta |
| `spectral_filtered_*` aggregation targets | `canonical_aggregation_target()` in `targets.py` | input alias only; outputs use `graph_filtered_*` |
| `ours_spectral_filtered_*` in CSV/result tags | `reporting.py` legacy pair prefixes | reporting only; not suite launch tokens |
| `suite_summary.*`, `suite_rows.json` | `artifacts.py` short names | read-only when canonical `vision_suite_*` absent |
| `spectral_filter_gain_*` trace fields | diagnostics metrics | operator/math naming, not public rename debt |
| `experiments_current/` | local output root | gitignored; `outputs/` rename deferred |

## Completed Migrations

| Debt | Status |
|---|---|
| `graph_filter_strength` canonical key and CLI | done; Phase 2 stopped mirroring old key in new meta |
| `ours_graph_filtered_*` suite tokens | done |
| `_graph_filter_only` suffix | done |
| `graph_filtered_*` operator output labels | done |
| `graphfl_lab` package root | done; `spectral_fl` shim removed |
| `graph_vectors_for_graphfl` | done; old helper module removed |
| Gate 6 `general_*` / `spectral_*` public surfaces | done |

Optional future work (breaking, policy decision required):

```text
- remove configs/general path alias
- remove spectral_filtered_* and spectral_filter_strength input aliases
- remove reporting legacy pair prefixes when old result reuse ends
```

## Future Rename Rule

Migration-required surfaces:

```text
public CLI names
config paths
generated artifact names
external import paths
```

Rename steps:

```text
1. Add canonical name.
2. Keep old name as wrapper/alias only when historical reuse requires it.
3. Update active docs/configs.
4. Run full tests and diagnostic preflight.
5. Remove old name after internal references and documented run paths move.
6. Record removal in docs/removed-materials.md.
```

Generated cache:

```text
__pycache__/ can be deleted in a dedicated cleanup pass.
```
