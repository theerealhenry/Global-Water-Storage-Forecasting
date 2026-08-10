# Architectural Design Records

`ARCHITECTURE.md` is the binding blueprint. Every deviation from it — dropping/reordering a champion-ladder level, changing a validation tier's definition, adding a model family not already planned, reversing an external-data decision, changing submission-selection criteria, or anything contradicting a specific claim in `COMPETITIVE_ANALYSIS.md` or a Definition of Done in `PROJECT_PLAN.md` — gets a new ADR here. The rule is not "never deviate," it's **never deviate silently.** See `ARCHITECTURE.md` §0 for the change workflow.

## Categories

- **Data** — versioning, external dataset adoption, source/provenance changes
- **Validation** — masking simulator changes, split changes, new validation tiers
- **Modeling** — adopting/dropping a model family, MoE, correction techniques
- **Features** — new feature families, signature/state definitions changing
- **Deployment** — serving stack changes, packaging changes

## Process

1. Copy `template.md` to `000N-short-title.md` (next sequential number).
2. Fill it in completely — context, evidence, decision, alternatives, consequences, risks, validation.
3. Commit alongside (or immediately after) the change it documents.
4. Add it to the index below.

## Index

| ID | Title | Category | Status | Date |
|---|---|---|---|---|
| [0001](0001-state-reconstruction-architecture.md) | Adopt state-reconstruction architecture over direct regression | Modeling | Accepted | 2026-08-09 |
| [0002](0002-no-dvc-manifest-based-data-versioning.md) | No DVC — manifest-based data versioning | Data | Accepted | 2026-08-09 |
| [0003](0003-dependency-management-strategy.md) | Conda-for-interpreter + pip-for-packages, pinned version set | Deployment | Accepted | 2026-08-10 |
