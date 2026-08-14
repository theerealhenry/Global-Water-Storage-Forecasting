"""Config-driven feature registry — Project Phase 4 step 4.4.

Mirrors ``validation/scenarios.py``'s ``load_scenario``/``list_scenarios``
shape exactly (``docs/ARCHITECTURE.md`` §11's "assigned an identifier and
lives as a configuration file" discipline, applied here to feature
tunables instead of validation scenarios): every tunable Project Phase 4
introduces — shrinkage ``k``, neighbor count and distance-weighting choice,
trailing window lengths, SPEI-differencing lags, drought-run-length
thresholds — lives in a named ``configs/features/*.yaml`` file, referenced
by identifier, never a hardcoded Python constant scattered across feature
modules.

This module is the only code path that reads those files. It does not
implement any transformer itself — ``state/signatures.py`` (step 4.2),
``state/spatial_history.py`` (step 4.3), ``features/temporal.py`` (step
4.5), and ``features/environmental.py`` (step 4.6) each load their own
named config from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from tws_forecast.data.loaders import get_repo_root

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureConfig",
    "FEATURE_CONFIG_DIR",
    "FEATURE_CONFIG_REGISTRY",
    "list_feature_configs",
    "load_feature_config",
]

FEATURE_CONFIG_DIR = get_repo_root() / "configs" / "features"

FeatureType = Literal["signatures", "spatial_history", "temporal", "environmental"]


class FeatureConfig(BaseModel):
    """Validated contents of one ``configs/features/*.yaml`` file.

    ``feature_type`` discriminates which fields beyond the always-present
    ones (``name``, ``feature_type``, ``description``, ``source_rationale``)
    are required — each Project Phase 4 feature module reads exactly the
    subset relevant to it.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    feature_type: FeatureType
    description: str
    source_rationale: str

    # feature_type == "signatures" only
    shrinkage_k: int | None = None
    trailing_windows: tuple[int, ...] | None = None

    # feature_type == "spatial_history" only
    n_neighbors: int | None = None
    distance_weighting: Literal["inverse_distance", "flat_mean"] | None = None
    max_neighbor_distance_km: float | None = None

    # feature_type == "temporal" only
    trend_window_months: tuple[int, ...] | None = None

    # feature_type == "environmental" only
    spei_diff_lags: tuple[int, ...] | None = None
    drought_threshold: float | None = None

    @model_validator(mode="after")
    def _check_required_fields_for_type(self) -> FeatureConfig:
        required_by_type: dict[str, tuple[str, ...]] = {
            "signatures": ("shrinkage_k", "trailing_windows"),
            "spatial_history": ("n_neighbors", "distance_weighting"),
            "temporal": ("trend_window_months",),
            "environmental": ("spei_diff_lags", "drought_threshold"),
        }
        required = required_by_type[self.feature_type]
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(f"feature_type={self.feature_type!r} requires {missing}, got None")

        if self.feature_type == "signatures" and self.shrinkage_k is not None:
            if self.shrinkage_k <= 0:
                raise ValueError(f"shrinkage_k must be > 0, got {self.shrinkage_k}")

        if self.feature_type == "spatial_history" and self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(f"n_neighbors must be > 0, got {self.n_neighbors}")

        return self


def _discover_feature_configs() -> dict[str, Path]:
    if not FEATURE_CONFIG_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(FEATURE_CONFIG_DIR.glob("*.yaml"))}


FEATURE_CONFIG_REGISTRY: dict[str, Path] = _discover_feature_configs()


def list_feature_configs() -> list[str]:
    """Names of every registered feature config, sorted."""
    return sorted(FEATURE_CONFIG_REGISTRY)


def load_feature_config(name: str) -> FeatureConfig:
    """Load and validate one named feature config from
    ``configs/features/``.

    Parameters
    ----------
    name:
        A feature-config identifier — the YAML filename's stem, e.g.
        ``"signatures"``. Use :func:`list_feature_configs` to see what's
        registered.
    """
    if name not in FEATURE_CONFIG_REGISTRY:
        raise KeyError(f"No feature config named {name!r}. Available: {list_feature_configs()}")

    path = FEATURE_CONFIG_REGISTRY[name]
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = FeatureConfig.model_validate(raw)
    if config.name != name:
        raise ValueError(
            f"Feature config file {path.name} declares name={config.name!r}, which "
            f"does not match its filename stem {name!r} — the registry key "
            "and the declared name must agree."
        )

    logger.info("Loaded feature config %r (type=%s)", name, config.feature_type)
    return config
