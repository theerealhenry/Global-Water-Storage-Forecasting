"""Centralized random-seed control.

Every stochastic operation added from Project Phase 2 onward — resampling in
the masking simulator, CV-fold construction where any randomness is
involved, later model training — takes an explicit seed, defaulting to
``RANDOM_SEED``, never an unseeded call. This is required for the project's
own computational-reproducibility guarantee (``docs/ARCHITECTURE.md`` §18)
and because the competition re-runs top-10 finishers' code and requires
fixed seeds (``docs/ARCHITECTURE.md`` §1).
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RANDOM_SEED", "set_seed"]

RANDOM_SEED: int = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every global stochastic source this project currently uses.

    Seeds Python's ``random`` module and numpy's global RNG. Model-specific
    seeding (LightGBM's ``random_state``, XGBoost's ``seed``, etc., added in
    Project Phase 5) is passed explicitly at model-construction time in
    those later modules — those libraries don't share a global RNG with
    numpy/random, so seeding them here would be a no-op; this function only
    seeds the sources that genuinely are global.

    Parameters
    ----------
    seed:
        Defaults to the project-wide ``RANDOM_SEED`` constant. Callers that
        need a specific, different seed (e.g. a robustness check across
        several seeds near a promotion decision, per
        ``docs/ARCHITECTURE.md`` §11) pass it explicitly.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeded random and numpy with seed=%d", seed)
