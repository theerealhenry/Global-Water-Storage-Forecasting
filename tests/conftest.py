"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    """Directory containing the small, committed golden test fixtures."""
    return Path(__file__).resolve().parent / "data" / "golden"
