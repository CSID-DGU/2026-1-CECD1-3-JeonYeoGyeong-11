# Removed Materials And Tombstones

This document records material removed or archived during the GraphFL Lab
cleanup/rename. It is also the long-term pointer for closed maintenance
documents after Gate 6.

## Release Anchors

| Anchor | Status | Notes |
|---|---|---|
| `pre-graphfl-rename` | created | SHA `e647da931bb3a78cc228ac2ad31103537b5ed640`; Gate 0 workspace baseline before Gate 1 inventory. |

## Tombstones

| Path or material | Status | Replacement |
|---|---|---|
| `spectral_fl/__init__.py` (package shim) | removed Gate 6 batch 6 | `graphfl_lab` canonical imports; serialized assets scanned via `scripts/dev/migrate_serialized_objects.py` |
| `docs/maintenance/cleanup-status.md` | active until Gate 6 | After Gate 6, mark closed and link from this table. |
| `docs/framework/experimental-design.md` | removed duplicate bridge | `docs/framework/graph_fl_experimental_design.md`, `docs/framework/graph_fl_experimental_design_appendix.md` |

## Archive Policy

Current project docs should stay in `docs/framework/`, `docs/research/`, and
root-level docs. Prior directions and superseded material move to `docs/archive/`
or are represented by a tombstone here with the relevant tag SHA.
