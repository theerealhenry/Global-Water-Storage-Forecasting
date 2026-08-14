"""Historical spatial-neighbor features — Project Phase 4 step 4.3.

Every feature this module produces is **S2 (historical) or S3 (signature)**
per the S1-S4 taxonomy in ``docs/ARCHITECTURE.md`` §9 — never S1
(concurrent, same-month neighbor values). Same-month neighbors are almost
always masked *together* during a real blackout (the entire grid loses its
current observation at once), so a same-month feature would be unusable in
exactly the regime this project's masking structure makes common — the
verified 0.981-correlation-but-unusable-during-blackouts finding
(``docs/PROJECT_PLAN.md`` "Key findings"). :data:`SPATIAL_FEATURE_TAXONOMY`
records this classification per feature name so it is a mechanically
checked invariant, not just a documentation promise.

This module composes step 4.1 (``state.reconstruction`` — ``last_known``,
via ``build_state_snapshots``) and step 4.2 (``state.signatures`` —
shrinkage-regularized trend/seasonality/ACF, via
``compute_location_signatures``) rather than recomputing either kind of
per-location logic independently for neighbors: a neighbor's own state and
signature are built exactly the same way this location's own are, then
joined and aggregated across the k nearest neighbors.

Neighbors are found by great-circle (haversine) distance on ``(lat, lon)``
via a ``sklearn.neighbors.BallTree`` — an exact, standard k-NN structure
that avoids the O(n^2) pairwise-distance matrix a naive implementation
would build (infeasible at this project's real ~15,715-location grid).
``n_neighbors``, ``distance_weighting``, and ``max_neighbor_distance_km``
are config-driven (``configs/features/spatial_history.yaml``, step 4.4's
registry), not hardcoded. No basin dataset is sourced as of this phase —
``docs/PROJECT_PLAN.md``'s "basin-aware aggregation if a basin dataset is
sourced" is an explicitly conditional, documented extension point, not
built here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from tws_forecast.features.registry import load_feature_config
from tws_forecast.state.reconstruction import build_state_snapshots, ensure_location_id
from tws_forecast.state.signatures import compute_location_signatures

__all__ = [
    "SPATIAL_FEATURE_TAXONOMY",
    "EARTH_RADIUS_KM",
    "SpatialHistoryTransformer",
]

SCategory = Literal["S1", "S2", "S3", "S4"]

#: Every feature column this module can produce, tagged with its
#: ``ARCHITECTURE.md`` §9 category — a standing, mechanical guard (see
#: ``tests/test_spatial_history.py::test_no_s1_feature_in_taxonomy``)
#: against ever silently reintroducing a same-month (S1) neighbor feature.
SPATIAL_FEATURE_TAXONOMY: dict[str, SCategory] = {
    "neighbor_TWS_last_known": "S2",
    "neighbor_TWS_lag_3": "S2",
    "neighbor_TWS_lag_6": "S2",
    "neighbor_historical_anomaly": "S2",
    "neighbor_trend": "S3",
    "neighbor_seasonal_signature": "S3",
    "neighbor_ACF": "S3",
}

EARTH_RADIUS_KM = 6371.0


def _resolve_spatial_config(
    n_neighbors: int | None,
    distance_weighting: str | None,
    max_neighbor_distance_km: float | None,
) -> tuple[int, str, float]:
    if (
        n_neighbors is not None
        and distance_weighting is not None
        and max_neighbor_distance_km is not None
    ):
        return n_neighbors, distance_weighting, max_neighbor_distance_km
    config = load_feature_config("spatial_history")
    return (
        n_neighbors if n_neighbors is not None else config.n_neighbors,
        distance_weighting if distance_weighting is not None else config.distance_weighting,
        (
            max_neighbor_distance_km
            if max_neighbor_distance_km is not None
            else config.max_neighbor_distance_km
        ),
    )


def _build_neighbor_edges(
    unique_locations: pd.DataFrame,
    n_neighbors: int,
    distance_weighting: str,
    max_neighbor_distance_km: float,
) -> pd.DataFrame:
    """One row per (location_id, neighbor_location_id, weight) edge, built
    once at ``fit()`` time via an exact haversine k-NN query (never an
    O(n^2) pairwise distance matrix, which would not scale to this
    project's real ~15,715-location grid)."""
    if len(unique_locations) < 2:
        return pd.DataFrame(
            columns=["location_id", "neighbor_location_id", "distance_km", "weight"]
        )

    coords_rad = np.radians(unique_locations[["lat", "lon"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    k = min(n_neighbors + 1, len(unique_locations))  # +1: query includes the point itself
    distances_rad, indices = tree.query(coords_rad, k=k)

    location_ids = unique_locations["location_id"].to_numpy()
    edges: list[dict[str, object]] = []
    for i, loc_id in enumerate(location_ids):
        candidates = []
        for dist_rad, j in zip(distances_rad[i], indices[i], strict=True):
            if j == i:
                continue
            distance_km = float(dist_rad) * EARTH_RADIUS_KM
            if distance_km <= max_neighbor_distance_km:
                candidates.append((location_ids[j], distance_km))
        candidates = candidates[:n_neighbors]
        for neighbor_id, distance_km in candidates:
            if distance_weighting == "inverse_distance":
                weight = 1.0 / max(distance_km, 1e-6)
            else:
                weight = 1.0
            edges.append(
                {
                    "location_id": loc_id,
                    "neighbor_location_id": neighbor_id,
                    "distance_km": distance_km,
                    "weight": weight,
                }
            )

    if not edges:
        return pd.DataFrame(
            columns=["location_id", "neighbor_location_id", "distance_km", "weight"]
        )
    return pd.DataFrame(edges)


class SpatialHistoryTransformer:
    """``features.base.Transformer`` producing the k-NN historical spatial
    features named in :data:`SPATIAL_FEATURE_TAXONOMY`.

    ``fit(train_df)`` builds the k-NN neighbor index (haversine distance on
    each location's fixed ``(lat, lon)``) and stores the training frame as
    historical context, mirroring ``state.signatures.LocationSignatureTransformer``.
    ``transform(df)`` composes ``build_state_snapshots`` and
    ``compute_location_signatures`` over the training-plus-``df`` history to
    get every neighbor's own last-known value and shrinkage-regularized
    signature *at the same origin time* as each row of ``df``, then
    aggregates across each row's k nearest neighbors with the configured
    distance weighting.

    A location with zero neighbors within ``max_neighbor_distance_km``
    (edge-of-grid or an isolated point) falls back to a global, all-location
    pooled estimate for every feature — never ``NaN`` and never a raised
    exception.
    """

    def __init__(
        self,
        n_neighbors: int | None = None,
        distance_weighting: str | None = None,
        max_neighbor_distance_km: float | None = None,
        shrinkage_k: int | None = None,
    ) -> None:
        self._n_neighbors, self._distance_weighting, self._max_neighbor_distance_km = (
            _resolve_spatial_config(n_neighbors, distance_weighting, max_neighbor_distance_km)
        )
        self._shrinkage_k = shrinkage_k
        self._train_df: pd.DataFrame | None = None
        self._neighbor_edges: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df.copy()
        frame = ensure_location_id(train_df)
        unique_locations = frame.drop_duplicates("location_id")[["location_id", "lat", "lon"]]
        self._neighbor_edges = _build_neighbor_edges(
            unique_locations,
            self._n_neighbors,
            self._distance_weighting,
            self._max_neighbor_distance_km,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._train_df is None or self._neighbor_edges is None:
            raise RuntimeError("SpatialHistoryTransformer.transform called before fit()")

        frame = ensure_location_id(df).copy()
        frame["time"] = pd.to_datetime(frame["time"])
        frame["period"] = pd.PeriodIndex(frame["time"], freq="M")

        combined = pd.concat([self._train_df, df], ignore_index=False)
        combined = ensure_location_id(combined).copy()
        combined["time"] = pd.to_datetime(combined["time"])
        # Deduplicate by (location_id, time) content, never by raw pandas
        # index -- see state.signatures.LocationSignatureTransformer.transform's
        # identical fix for why index-based deduplication is unsafe here.
        combined = combined.loc[~combined.duplicated(subset=["location_id", "time"], keep="last")]
        combined["period"] = pd.PeriodIndex(combined["time"], freq="M")

        # Every neighbor's own state/signature, at every period any row in
        # this frame might need -- computed once over the whole panel, via
        # the already-vectorized step 4.1/4.2 machinery, never recomputed
        # per neighbor lookup.
        state_panel = build_state_snapshots(combined, as_of_column="time")
        state_panel["period"] = pd.PeriodIndex(state_panel["as_of"], freq="M")
        state_lookup = state_panel[["location_id", "period", "last_known_tws"]].rename(
            columns={
                "location_id": "neighbor_location_id",
                "last_known_tws": "neighbor_TWS_last_known",
            }
        )

        signature_panel = compute_location_signatures(
            combined, as_of_column="time", shrinkage_k=self._shrinkage_k
        )
        signature_panel["period"] = pd.PeriodIndex(signature_panel["as_of"], freq="M")
        signature_lookup = signature_panel[
            ["location_id", "period", "mean", "trend", "seasonality_amplitude", "acf_1"]
        ].rename(
            columns={
                "location_id": "neighbor_location_id",
                "mean": "neighbor_signature_mean",
                "trend": "neighbor_trend",
                "seasonality_amplitude": "neighbor_seasonal_signature",
                "acf_1": "neighbor_ACF",
            }
        )

        calendar_lookup = combined[["location_id", "period", "TWS_t"]].rename(
            columns={"location_id": "neighbor_location_id"}
        )

        frame_reset = frame.reset_index().rename(columns={"index": "_row_idx"})[
            ["_row_idx", "location_id", "period"]
        ]
        pairs = frame_reset.merge(self._neighbor_edges, on="location_id", how="left")

        pairs = pairs.merge(state_lookup, on=["neighbor_location_id", "period"], how="left")
        pairs = pairs.merge(signature_lookup, on=["neighbor_location_id", "period"], how="left")
        pairs["neighbor_historical_anomaly"] = (
            pairs["neighbor_TWS_last_known"] - pairs["neighbor_signature_mean"]
        )

        pairs["lag3_period"] = pairs["period"] - 3
        lag3_lookup = calendar_lookup.rename(
            columns={"period": "lag3_period", "TWS_t": "neighbor_TWS_lag_3"}
        )
        pairs = pairs.merge(lag3_lookup, on=["neighbor_location_id", "lag3_period"], how="left")

        pairs["lag6_period"] = pairs["period"] - 6
        lag6_lookup = calendar_lookup.rename(
            columns={"period": "lag6_period", "TWS_t": "neighbor_TWS_lag_6"}
        )
        pairs = pairs.merge(lag6_lookup, on=["neighbor_location_id", "lag6_period"], how="left")

        feature_cols = list(SPATIAL_FEATURE_TAXONOMY.keys())
        for col in feature_cols:
            valid = pairs[col].notna() & pairs["weight"].notna()
            pairs[f"_w_{col}"] = np.where(valid, pairs["weight"], 0.0)
            pairs[f"_wv_{col}"] = np.where(valid, pairs["weight"] * pairs[col], 0.0)

        sums = pairs.groupby("_row_idx")[
            [f"_wv_{c}" for c in feature_cols] + [f"_w_{c}" for c in feature_cols]
        ].sum()

        aggregated = pd.DataFrame(index=sums.index)
        for col in feature_cols:
            w_sum = sums[f"_w_{col}"]
            with np.errstate(invalid="ignore", divide="ignore"):
                aggregated[col] = sums[f"_wv_{col}"] / w_sum.replace(0.0, np.nan)

        fallback = _global_fallback(combined)
        for col in feature_cols:
            aggregated[col] = aggregated[col].fillna(fallback[col])

        aggregated = aggregated.reindex(frame_reset["_row_idx"])
        aggregated.index = frame.index
        return aggregated[feature_cols]


def _global_fallback(combined: pd.DataFrame) -> dict[str, float]:
    """A neutral, all-location fallback for locations with zero neighbors
    within ``max_neighbor_distance_km`` -- deliberately coarse (a single,
    non-time-varying pooled estimate, not itself origin-time-indexed): this
    path only exists for edge-of-grid/isolated locations where no real
    spatial signal is available anyway, so a simple neutral prior is
    preferable to raising or returning ``NaN`` into a downstream model."""
    observed = combined.dropna(subset=["TWS_t"])
    overall_mean = float(observed["TWS_t"].mean()) if not observed.empty else 0.0
    return {
        "neighbor_TWS_last_known": overall_mean,
        "neighbor_TWS_lag_3": overall_mean,
        "neighbor_TWS_lag_6": overall_mean,
        "neighbor_historical_anomaly": 0.0,
        "neighbor_trend": 0.0,
        "neighbor_seasonal_signature": 0.0,
        "neighbor_ACF": 0.0,
    }
