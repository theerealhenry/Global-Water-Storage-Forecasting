# ADR-0005 — Reconcile Phase 2 module naming with `ARCHITECTURE.md`'s canonical map

**Status:** Accepted
**Date:** 2026-08-11
**Category:** Validation
**Deciders:** Steve, Claude

## Context

Before starting implementation, Phase 2 (validation harness) was reviewed end-to-end against all governing
documents, per `ARCHITECTURE.md` §2's own precedence order: competition rules, then `ARCHITECTURE.md`
itself, then `PROJECT_PLAN.md`, then `COMPETITIVE_ANALYSIS.md`. That review surfaced a real inconsistency
that would otherwise have been carried silently into code.

`ARCHITECTURE.md` §6 (repository/module map) and §11 (validation engine) specify, in prose written before
any Phase 2 code existed, a specific set of files under `src/tws_forecast/validation/`: `splitters.py`,
`masking_simulator.py` (built around `MaskingScenario` configuration objects), `scenarios.py` (a
config-driven scenario registry, backed by `configs/validation/*.yaml`), `tiers.py`, `decomposition.py`,
and `leakage_tests.py`. §4 additionally specifies a `ForecastOrigin` schema (fields `origin_time,
target_time, horizon, information_cutoff, location_id, regime`), living in `state/reconstruction.py`
alongside the (Phase-4-scoped) `StateSnapshot`, as the join key every validation and feature artifact keys
off.

`PROJECT_PLAN.md`'s Phase 2 section, written and expanded during the Phase 1 closure work (ADR-0004,
2026-08-11), independently specified a Phase 2 build plan using different names — `splits.py` (not
`splitters.py`), no `scenarios.py`/config registry, no explicit `ForecastOrigin` construction step, and a
`harness.py` not present in `ARCHITECTURE.md`'s file list at all. ADR-0004 bound Phase 2's *numerical
constants* (the k-distribution, calendar offsets, baseline values) tightly to Phase 1's measured evidence,
which remains entirely correct and unaffected by this ADR — but it did not check its own file/module names
against `ARCHITECTURE.md`, and in doing so introduced exactly the kind of silent architectural drift §2
of `ARCHITECTURE.md` says must never happen without a recorded decision.

## Evidence

- `ARCHITECTURE.md` line 140-142 (module map): `validation/ splitters.py, masking_simulator.py
  (MaskingScenario objects), scenarios.py (registry, config-driven), tiers.py, decomposition.py,
  leakage_tests.py`.
- `ARCHITECTURE.md` §11 describes the three-tier design, the `MaskingScenario` config contract (§8:
  `blackout_start, blackout_end, affected_locations, exception_rate, streak_length, transition_pattern,
  source/rationale`), and named, config-file-backed scenarios ("a standard expanding-window scenario, a
  seven-month synthetic-blackout scenario, a scenario approximating 2015-like conditions, a scenario
  approximating a broader drought regime") referenced by identifier rather than re-described inline —
  none of this config-driven registry concept appears in `PROJECT_PLAN.md`'s Phase 2 section as written.
- `ARCHITECTURE.md` §4 requires `ForecastOrigin` as the system's origin/cutoff enforcement mechanism,
  used by "every other artifact in the system — state snapshots, features, out-of-fold predictions."
  `PROJECT_PLAN.md`'s Phase 2 section never mentions building it, which would leave the validation engine
  without the join key `ARCHITECTURE.md` says it depends on.
- `ARCHITECTURE.md` §11's leakage-firewall checks ("a future-row shuffle test," "a historical-only check,"
  "rolling-window features are verified to stop exactly at the information cutoff," "the masking simulator
  is checked to confirm it cannot leak a value it has just hidden") are named as mechanical, code-level
  checks belonging to a `leakage_tests.py` module — `PROJECT_PLAN.md`'s Phase 2 section folds an
  equivalent idea into a single bullet (§2.4) without naming the module.

## Current architecture

No Phase 2 code exists yet (confirmed: `src/tws_forecast/validation/__init__.py` is an empty stub). This
is exactly the moment to fix a naming/scope mismatch — before any file is written under either scheme —
rather than after, when renaming would touch working code, tests, and config references.

## Decision

`ARCHITECTURE.md` is authoritative on module structure, per its own stated precedence order. Phase 2 is
implemented against `ARCHITECTURE.md`'s file map, not `PROJECT_PLAN.md`'s. Specifically:

1. **`src/tws_forecast/state/reconstruction.py`** is created in Phase 2 (not deferred entirely to Phase 4)
   containing only the `ForecastOrigin` dataclass for now, with a docstring noting `StateSnapshot` will be
   added to the same file in Phase 4 per the architecture's stated co-location. This is a narrow,
   deliberate exception to "Phase 2 builds only validation": `ForecastOrigin` is infrastructure the
   validation engine cannot function without, and `ARCHITECTURE.md` §5 places it in layer 3 (Forecast
   Information Set), directly below layer 6 (Validation Engine) — it has no other natural home.
2. **`src/tws_forecast/validation/splitters.py`** (not `splits.py`) implements the expanding-window split
   generator, producing `ForecastOrigin`-indexed folds.
3. **`src/tws_forecast/validation/masking_simulator.py`** implements a `MaskingScenario` dataclass
   (fields per `ARCHITECTURE.md` §8: `blackout_start, blackout_end, affected_locations, exception_rate,
   streak_length, transition_pattern, source_rationale`) plus the streak-aware simulator function that
   consumes it. The two named modes from `PROJECT_PLAN.md` §2.2 (`curve`, replaying Experiment 3;
   `replay`, replaying Experiment 4's Method B) become two pre-built `MaskingScenario` instances in the
   registry below, not two code-level branches inside the simulator itself — this keeps the simulator
   generic and the specific real-world scenarios declarative.
4. **`src/tws_forecast/validation/scenarios.py`** is added (new relative to `PROJECT_PLAN.md`'s prior Phase
   2 section) as a config-driven registry, backed by `configs/validation/*.yaml` — one file per named
   scenario (`expanding_window.yaml`, `blackout_curve_k9.yaml`, `test_regime_replay.yaml`, at minimum).
   Experiments and tiers reference a scenario by identifier, never re-describe split/masking logic inline,
   per `ARCHITECTURE.md` §11's explicit requirement.
5. **`src/tws_forecast/validation/tiers.py`** implements Tier 1/2/3 as three functions or a small class
   hierarchy consuming `splitters.py` + `scenarios.py`, each returning predictions/scores in the shape
   `decomposition.py` expects.
6. **`src/tws_forecast/validation/decomposition.py`** implements the error-decomposition table and the
   degradation-slope metric, using the real k=2-7 buckets and ACF-quartile cross-cut from
   `phase1_constants.py` (retained from ADR-0004, unaffected by this ADR).
7. **`src/tws_forecast/validation/leakage_tests.py`** implements the four mechanical checks named in
   `ARCHITECTURE.md` §7/§11 as reusable, importable check functions (not only as pytest test functions),
   so they can also be invoked as a runtime assertion inside `harness.py` if ever useful — `tests/` then
   contains thin pytest wrappers that call into this module, keeping the actual check logic in `src/`
   where the rest of the module map expects it.
8. **`src/tws_forecast/validation/harness.py` is kept**, as an addition to `ARCHITECTURE.md`'s file list
   rather than a replacement of anything in it — `ARCHITECTURE.md` describes an orchestration
   responsibility ("final model selection is always made against Tier 1 and Tier 2," "a candidate ... is
   only promoted ... when it improves the relevant validation tier without materially degrading another")
   that needs one composition point tying `splitters.py` + `scenarios.py` + `tiers.py` +
   `decomposition.py` together and enforcing the integrity safeguard from `PROJECT_PLAN.md` §2.4. This is
   a genuine, narrow addition to the architecture, recorded here rather than introduced silently.
9. **`src/tws_forecast/validation/phase1_constants.py` is kept exactly as ADR-0004 specified** — it isn't
   named in `ARCHITECTURE.md`'s module map (written before Phase 1's numbers existed to name), and nothing
   about it conflicts with the architecture; it is a pure addition, not a naming conflict, so it needs no
   reconciliation.

## Reason

`ARCHITECTURE.md` states its own precedence explicitly: "Code is never allowed to diverge silently from
this document, and this document is never allowed to diverge silently from what an experiment actually
showed." Neither condition applied here — no code exists, and no experiment contradicted the architecture
— so the correct resolution is to make the plan match the architecture, not to update the architecture to
match a plan drafted without cross-checking it. Building against `splits.py`/no-scenario-registry would
have meant re-deriving, under a different name and without the config-driven registry, something
`ARCHITECTURE.md` had already specified more completely.

## Alternatives considered

**Update `ARCHITECTURE.md` to match `PROJECT_PLAN.md`'s simpler naming** — rejected: `ARCHITECTURE.md`'s
version is more complete (it specifies the config-driven scenario registry and `ForecastOrigin`, both of
which are real, load-bearing pieces of the system's stated design, not incidental naming choices), and
`ARCHITECTURE.md` is the higher-authority document by the project's own governance rule. Simplifying the
architecture down to match an under-specified plan would remove real design content, not just rename files.

**Leave both documents as-is and let implementation pick whichever it prefers** — rejected: this is
exactly the "quiet drift" `ARCHITECTURE.md` §2 exists to prevent, and would leave a permanent inconsistency
between the two documents for anyone reading them later (including a competition judge or a portfolio
reviewer).

## Consequences

`docs/PHASE2_EXECUTION_PLAN.md` (written immediately after this ADR) is built against the module list in
the Decision section above. `PROJECT_PLAN.md`'s Phase 2 section is trimmed to a short summary pointing to
that document, rather than maintaining two independently-evolving descriptions of the same phase.

## Risks

None beyond ordinary file-naming risk, fully mitigated by making this decision before any file is written
under the old scheme.

## Validation

Confirmed correct if: `src/tws_forecast/validation/` and `src/tws_forecast/state/reconstruction.py`, once
built, match this ADR's file list exactly, and `ARCHITECTURE.md` §6/§11 require no further correction once
Phase 2 is complete.

## Affected components

- [x] validation
- [x] data
- [ ] modeling
- [ ] features
- [ ] deployment
- [x] documentation

## Related

- Experiments: none (architectural/governance decision, not evidence-driven)
- MLflow runs: none
- Submissions: none
- Supersedes: the file-naming portions of ADR-0004's Decision items 1-2 (the numerical-constants binding
  in ADR-0004 is fully retained and unaffected)
- Superseded by: none
- Related ADRs: ADR-0001 (state-reconstruction architecture — this ADR's `ForecastOrigin`-in-Phase-2
  decision is a direct consequence of ADR-0001's layering), ADR-0004 (numerical constants this ADR's
  modules must still honor)

## Follow-up actions

- [x] `docs/PHASE2_EXECUTION_PLAN.md` written against this ADR's module list
- [x] `PROJECT_PLAN.md` Phase 2 section trimmed to point to it
- [x] `ARCHITECTURE.md` §20 status note updated once Phase 2 implementation begins — updated again at
  Phase 2's formal close (2026-08-13) to record step 2.11's proof-run results
