"""Minimal project configuration loader.

Deliberately small: this reads ``configs/base.yaml`` into a validated
pydantic model. Full config-driven modeling configuration (feature sets,
model hyperparameters, ``champion.yaml``) arrives in later phases as those
pieces of the system are built (``docs/ARCHITECTURE.md`` §6) — this module
exists now because Project Phase 2's scenario registry
(``validation/scenarios.py``, step 2.5) needs a way to resolve
``configs/validation/*.yaml`` paths and the shared ``random_seed``/
``data_dir`` values without each caller hardcoding them.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from tws_forecast.data.loaders import get_repo_root

logger = logging.getLogger(__name__)

__all__ = ["Period", "BaseConfig", "load_base_config", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = get_repo_root() / "configs" / "base.yaml"


class Period(BaseModel):
    """A closed date interval, e.g. the training or test period."""

    start: date
    end: date


class BaseConfig(BaseModel):
    """Validated contents of ``configs/base.yaml``."""

    random_seed: int
    data_dir: str
    train_period: Period
    test_period: Period


def load_base_config(path: Path | str | None = None) -> BaseConfig:
    """Load and validate ``configs/base.yaml`` (or an explicit override path).

    Parameters
    ----------
    path:
        Defaults to ``DEFAULT_CONFIG_PATH`` (``<repo_root>/configs/base.yaml``).
        Overridable so tests can point at a fixture config instead.
    """
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Base config not found at {resolved}")

    with open(resolved) as f:
        raw = yaml.safe_load(f)

    config = BaseConfig.model_validate(raw)
    logger.info("Loaded base config from %s (seed=%d)", resolved, config.random_seed)
    return config
