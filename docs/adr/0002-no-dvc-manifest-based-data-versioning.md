# ADR-0002 — No DVC — manifest-based data versioning

**Status:** Accepted
**Date:** 2026-08-09
**Category:** Data
**Deciders:** Steve, Claude

## Context
`PROJECT_PLAN.md` Project Phase 0 originally left large-file/data versioning as an open decision, given Train.csv is ~290MB. A senior-level MLOps portfolio project conventionally reaches for DVC here.

## Evidence
The dataset is externally supplied by Zindi, static for the competition's duration, single-version, and not something we redistribute. No evidence exists of multiple dataset versions or a need to reproduce data changes over time.

## Current architecture
No data-versioning mechanism existed prior to this decision.

## Decision
Use `data/raw/` (gitignored) + `data/README.md` (access/download instructions) + `dataset_manifest.json` (SHA256 hash and row count per file) instead of DVC.

## Reason
A hash-verified manifest gives the core reproducibility guarantee we actually need (anyone reproducing the repo can confirm byte-identical data) without DVC's setup and maintenance overhead, which doesn't buy proportional value for a static, single-version, non-redistributed dataset.

## Alternatives considered
**DVC with a local or cloud remote** — the conventional/expected choice, genuinely justified when a dataset changes over time, needs redistribution, or has multiple concurrent versions. None of those apply here.

## Consequences
`PROJECT_PLAN.md` Project Phase 0's data-handling decision is resolved, not open. Removed from the open-decisions list.

## Risks
If the dataset situation changes (e.g. Zindi issues a corrected Test.csv mid-competition, or we start versioning multiple processed feature tables that need their own history), this decision should be revisited — not assumed to still hold indefinitely.

## Validation
Correct if, throughout the project, no situation arises that actually required DVC's capabilities (branching/versioning multiple data states). If one does, that's the trigger for a superseding ADR.

## Affected components
- [x] data
- [x] documentation
- [ ] validation
- [ ] features
- [ ] modeling
- [ ] deployment

## Related
- Experiments: none
- MLflow runs: none
- Submissions: none
- Supersedes: none

## Follow-up actions
`PROJECT_PLAN.md` Project Phase 0 and its open-decisions section already reflect this as resolved. `ARCHITECTURE.md` §4 documents the resulting data architecture. No further edit needed unless this decision is later superseded.
