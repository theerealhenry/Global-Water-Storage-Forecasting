# ADR-0006 — Extend `StateSnapshot` with `second_previous_known_tws` and `state_acceleration`

**Status:** Accepted
**Date:** 2026-08-14
**Category:** Features
**Deciders:** Steve, Claude

## Context

Project Phase 4 (state reconstruction & feature engineering) begins with step 4.1: build the
`StateSnapshot` schema `docs/ARCHITECTURE.md` §4 assigns to `src/tws_forecast/state/reconstruction.py`.
Before writing any code, `docs/PHASE4_EXECUTION_PLAN.md` §0/§4.1 flagged a real, unresolved gap between two
governing documents rather than silently picking one.

`ARCHITECTURE.md` §4's `StateSnapshot` field list is: `last_known_tws, last_known_time,
months_since_observation, previous_known_tws, historical_delta, local_trend, seasonal_position,
acf_1_3_6_12, observation_density, blackout_streak_length, location_signature, state_status`. This gives
exactly two trajectory points (`last_known_tws`, `previous_known_tws`) and one derived quantity
(`historical_delta`, i.e. velocity).

`PROJECT_PLAN.md`'s Phase 4 section (written independently, before Phase 4 implementation began) asks for
"observation trajectory (last_known, previous_known, second_previous_known, from which state velocity **and
acceleration** are derived)" — three trajectory points, with acceleration as an explicit second derived
quantity. Acceleration is mathematically undefined from only two points; a third point
(`second_previous_known_tws`) is required to compute it as a second difference.

Per `ARCHITECTURE.md` §2's governance rule (architecture ranks above the project plan; evidence or a
genuine spec gap that contradicts a governing document triggers an ADR, never a silent pivot), this
discrepancy is resolved here before any `StateSnapshot` code is written, rather than one document being
followed and the other silently ignored.

## Evidence

- `ARCHITECTURE.md` §4 (`StateSnapshot` field list, quoted above) — no `second_previous_known_tws` or
  `state_acceleration` field.
- `PROJECT_PLAN.md`, Project Phase 4 section, first bullet: "observation trajectory (last_known,
  previous_known, second_previous_known, from which state velocity and acceleration are derived)."
- `docs/PHASE4_EXECUTION_PLAN.md` §4.1, written during Phase 4 planning specifically to surface this gap
  before implementation rather than during it.

## Current architecture

No `StateSnapshot` code exists yet — `state/reconstruction.py` currently implements only `ForecastOrigin`
(Project Phase 2), with an explicit docstring note that `StateSnapshot` is deliberately deferred to Phase 4.
This is the same situation ADR-0005 found itself resolving for Phase 2's module map: the moment to fix a
spec mismatch is before any code exists against either version, not after.

## Decision

Extend `StateSnapshot` by exactly two fields beyond `ARCHITECTURE.md` §4's list:

- `second_previous_known_tws: float | None` — the third most recent actually-observed TWS value at or
  before the snapshot's `as_of` time, for the snapshot's location.
- `state_acceleration: float | None` — the second difference of the observation trajectory:
  `historical_delta - (previous_known_tws - second_previous_known_tws)`, i.e. the change in `historical_delta`
  itself between the two most recent observed-to-observed steps. `None` whenever fewer than three observed
  points exist in history, exactly mirroring how `historical_delta` is already `None` whenever fewer than two
  exist.

`historical_delta` is unchanged and continues to serve as state *velocity*
(`last_known_tws - previous_known_tws`) — it is not renamed or duplicated. This is a minimal, purely additive
extension: every existing `ARCHITECTURE.md` §4 field keeps its exact name and meaning.

## Reason

`historical_delta` already gives velocity; the only genuinely missing piece to satisfy `PROJECT_PLAN.md`'s
Phase 4 requirement is acceleration, and acceleration needs one additional trajectory point to be defined at
all. Adding exactly that one field (plus the trajectory point it depends on) is the smallest change that
reconciles both documents — a full redesign of the trajectory representation was not warranted by a gap this
narrow.

## Alternatives considered

- **Follow `ARCHITECTURE.md` literally and drop acceleration from Phase 4 entirely.** Rejected: acceleration
  is explicitly requested in `PROJECT_PLAN.md`'s Phase 4 bullet as one of the four distinct temporal
  quantities this phase must build (`docs/PHASE4_EXECUTION_PLAN.md` §0, "Four distinct temporal
  quantities") — silently dropping it would under-deliver against the project plan without a recorded
  reason.
- **Follow `PROJECT_PLAN.md` literally and redesign `StateSnapshot` around an arbitrary-length trajectory
  list (e.g. `trajectory: list[float]`) instead of named fields.** Rejected: `ARCHITECTURE.md` §4's fields
  are named and typed specifically so downstream consumers (features, the uncertainty architecture, the
  eventual deployment UI) can reference a stable field name rather than an index into a list whose length
  might change; an unbounded list also invites scope creep (why stop at three points?) that neither document
  actually asks for.
- **Compute acceleration lazily from `previous_known_tws` and a freshly-recomputed second-previous value,
  without storing `second_previous_known_tws` as its own field.** Rejected: this would make
  `state_acceleration` derivable but not independently inspectable/testable, and would require every
  consumer that wants acceleration to redo the same three-point lookup `StateSnapshot` is supposed to do
  once, canonically — the entire point of this schema (`ARCHITECTURE.md` §4: "no other module is permitted
  to compute its own competing notion").

## Consequences

`StateSnapshot` now has 14 fields instead of `ARCHITECTURE.md`'s original 12. `build_state_snapshot`/
`build_state_snapshots` (step 4.1) implement both new fields with the same `None`-when-insufficient-history
contract as `previous_known_tws`/`historical_delta`. No other module changes; `ForecastOrigin` is untouched.
Any future consumer that wants acceleration (e.g. a Phase 4/5 feature or the uncertainty architecture) can
read `state_acceleration` directly rather than re-deriving it.

## Risks

Minimal — this is an additive schema change to a not-yet-built schema, not a change to working code. The
only real risk is scope creep beyond these two fields under the same justification; this ADR deliberately
does not extend the trajectory further than the one point `PROJECT_PLAN.md` explicitly asks for.

## Validation

`tests/test_state_snapshot.py` pins the exact `None`-below-three-observations contract for
`second_previous_known_tws`/`state_acceleration`, and a positive-case test confirms `state_acceleration`'s
arithmetic against a hand-computed three-point fixture. If a later phase finds this trajectory depth
insufficient (e.g. Project Phase 5 wants a longer window), that is itself a new ADR, not a silent field
addition.

## Affected components

- [x] features
- [x] documentation
- [ ] data
- [ ] validation
- [ ] modeling
- [ ] deployment

## Related

- Experiments:
- MLflow runs:
- Submissions:
- Supersedes:
- Superseded by:
- Related ADRs: ADR-0005 (same "resolve a spec gap before writing code" discipline, applied to Phase 2's
  module map)

## Follow-up actions

`ARCHITECTURE.md` §4's `StateSnapshot` field list gets a follow-up edit noting the two additional fields and
pointing to this ADR, done alongside Phase 4's step 4.10 documentation closure (not this commit, to keep
this ADR's own commit scoped to the decision record itself, per the project's one-file-per-commit
convention) — tracked as an open item in `docs/PHASE4_EXECUTION_PLAN.md` §4.10.
