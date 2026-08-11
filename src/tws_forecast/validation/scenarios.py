"""Config-driven validation-scenario registry.

Every validation scenario used anywhere in this project is a named,
versioned YAML file under ``configs/validation/`` — referenced by
identifier, never re-described inline (``docs/ARCHITECTURE.md`` §11's
explicit requirement: "Each validation scenario used across the project is
assigned an identifier and lives as a configuration file... experiments
reference these scenario identifiers rather than re-describing the same
split logic inline each time.").

This module is the only code path that reads those files. It does not
itself run anything — ``validation/tiers.py`` (step 2.6) consumes a
``ScenarioConfig`` to actually drive ``splitters.expanding_window_splits``
and ``masking_simulator.apply_masking``.
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
    "SplitterParams",
    "ScenarioConfig",
    "SCENARIO_DIR",
    "SCENARIO_REGISTRY",
    "list_scenarios",
    "load_scenario",
]

SCENARIO_DIR = get_repo_root() / "configs" / "validation"

ScenarioType = Literal["expanding_window", "blackout_curve", "test_regime_replay"]


class SplitterParams(BaseModel):
    """Arguments forwarded to ``validation.splitters.expanding_window_splits``."""

    model_config = ConfigDict(frozen=True)

    n_folds: int = 5
    val_window_months: int = 6
    min_train_months: int = 84
    anchor_to_2004: bool = True


class ScenarioConfig(BaseModel):
    """Validated contents of one ``configs/validation/*.yaml`` file.

    ``scenario_type`` discriminates which fields are required beyond the
    always-present ones: ``blackout_curve`` needs ``k_distribution`` and
    ``n_windows``; ``test_regime_replay`` needs ``full_offsets``,
    ``blackout_offsets``, and ``blackout_k_by_offset``. ``expanding_window``
    needs neither — it's ``splitter`` params alone, no masking.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    tier: Literal[1, 2, 3]
    scenario_type: ScenarioType
    description: str
    source_rationale: str

    splitter: SplitterParams = SplitterParams()

    # scenario_type == "blackout_curve" only
    k_distribution: tuple[int, ...] | None = None
    n_windows: int | None = None

    # scenario_type == "test_regime_replay" only
    full_offsets: tuple[int, ...] | None = None
    blackout_offsets: tuple[int, ...] | None = None
    blackout_k_by_offset: dict[int, int] | None = None

    exception_rate: float = 0.0

    @model_validator(mode="after")
    def _check_required_fields_for_type(self) -> ScenarioConfig:
        if self.scenario_type == "blackout_curve":
            missing = [f for f in ("k_distribution", "n_windows") if getattr(self, f) is None]
            if missing:
                raise ValueError(f"scenario_type=blackout_curve requires {missing}, got None")

        if self.scenario_type == "test_regime_replay":
            required = ("full_offsets", "blackout_offsets", "blackout_k_by_offset")
            missing = [f for f in required if getattr(self, f) is None]
            if missing:
                raise ValueError(f"scenario_type=test_regime_replay requires {missing}, got None")
            offsets_with_k = set(self.blackout_offsets)  # type: ignore[union-attr]
            k_keys = set(self.blackout_k_by_offset)  # type: ignore[arg-type]
            if offsets_with_k != k_keys:
                raise ValueError(
                    "blackout_k_by_offset's keys must exactly match "
                    f"blackout_offsets: {offsets_with_k} != {k_keys}"
                )
            if set(self.full_offsets) & set(self.blackout_offsets):  # type: ignore[operator]
                raise ValueError("full_offsets and blackout_offsets must not overlap")

        if not (0.0 <= self.exception_rate < 1.0):
            raise ValueError(f"exception_rate must be in [0, 1), got {self.exception_rate}")

        return self


def _discover_scenarios() -> dict[str, Path]:
    if not SCENARIO_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(SCENARIO_DIR.glob("*.yaml"))}


SCENARIO_REGISTRY: dict[str, Path] = _discover_scenarios()


def list_scenarios() -> list[str]:
    """Names of every registered scenario, sorted."""
    return sorted(SCENARIO_REGISTRY)


def load_scenario(name: str) -> ScenarioConfig:
    """Load and validate one named scenario from ``configs/validation/``.

    Parameters
    ----------
    name:
        A scenario identifier — the YAML filename's stem, e.g.
        ``"expanding_window"``. Use ``list_scenarios()`` to see what's
        registered.
    """
    if name not in SCENARIO_REGISTRY:
        raise KeyError(f"No scenario named {name!r}. Available: {list_scenarios()}")

    path = SCENARIO_REGISTRY[name]
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = ScenarioConfig.model_validate(raw)
    if config.name != name:
        raise ValueError(
            f"Scenario file {path.name} declares name={config.name!r}, which "
            f"does not match its filename stem {name!r} — the registry key "
            "and the declared name must agree."
        )

    logger.info("Loaded scenario %r (tier %d, type=%s)", name, config.tier, config.scenario_type)
    return config
