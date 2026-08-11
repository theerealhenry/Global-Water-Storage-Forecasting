"""Tests for tws_forecast.utils.seeds."""

import numpy as np

from tws_forecast.utils.seeds import RANDOM_SEED, set_seed


def test_set_seed_is_deterministic() -> None:
    set_seed()
    a = np.random.rand(10)
    set_seed()
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_set_seed_default_matches_random_seed_constant() -> None:
    set_seed(RANDOM_SEED)
    a = np.random.rand(5)
    set_seed()  # relies on the default arg equalling RANDOM_SEED
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_different_seeds_diverge() -> None:
    set_seed(1)
    a = np.random.rand(10)
    set_seed(2)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)


def test_set_seed_also_seeds_python_random() -> None:
    import random

    set_seed(7)
    a = [random.random() for _ in range(5)]
    set_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b
