"""The ``Transformer`` protocol — Project Phase 4 step 4.4.

Deliberately mirrors ``validation.tiers.Predictor``'s ``fit``/``predict``
shape, with ``transform`` in place of ``predict``: every feature-producing
class Project Phase 4 builds (``state/signatures.py`` step 4.2,
``state/spatial_history.py`` step 4.3, ``features/temporal.py`` step 4.5,
``features/environmental.py`` step 4.6, ``features/targets.py`` step 4.7)
implements this protocol, so a future feature pipeline
(``features/assemble.py``, step 4.9, and eventually Project Phase 10's
``pipelines/train.py``) can compose an arbitrary list of transformers
uniformly — the same way ``validation/harness.py`` composes an arbitrary
``Predictor`` without special-casing which baseline or model it's holding.

``fit``/``transform`` is deliberately the same two-call shape scikit-learn
itself uses, not a coincidence: it is the shape that makes "fit only on the
training fold, then transform any fold" mechanically enforceable, which is
exactly the leakage-safety invariant ``docs/ARCHITECTURE.md`` §4/§7 requires
of every feature in this project (``docs/PHASE4_EXECUTION_PLAN.md`` §1,
"Leakage-safe transformers, fit-only-on-train").
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = ["Transformer"]


@runtime_checkable
class Transformer(Protocol):
    """The minimal interface every Project Phase 4 feature-producing class
    implements.

    ``fit`` learns whatever the transformer needs from a training frame
    (e.g. a shrinkage-regularized location signature, a k-NN neighbor
    index, a drought-threshold calibration) — never from ``transform``'s
    own input, which may be a validation or test fold. ``transform`` must
    be a pure function of its input frame plus whatever ``fit`` already
    learned: calling it twice on the same frame must return identical
    output, and calling it on a frame ``fit`` has never seen must not raise
    merely because that frame is unfamiliar (it should fall back
    gracefully — e.g. to a shrinkage-regularized global prior — the same
    contract ``state.signatures`` and ``state.spatial_history`` document
    for never-observed locations).

    Stateless transformers (e.g. a pure calendar-arithmetic feature that
    needs no training-fold statistics) implement ``fit`` as a no-op,
    exactly as stateless ``Predictor`` baselines already do in
    ``validation.tiers``.
    """

    def fit(self, train_df: pd.DataFrame) -> None: ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
