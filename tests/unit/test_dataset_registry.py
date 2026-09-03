"""Unit tests for the dataset registry."""

from __future__ import annotations

import pytest

from vla_eval.core.exceptions import DatasetNotFoundError
from vla_eval.data.registry import get_dataset_spec, list_datasets


def test_list_datasets_returns_curated_defaults() -> None:
    datasets = list_datasets()
    names = {d.name for d in datasets}
    assert "pusht" in names
    assert "aloha_sim_insertion_human" in names
    assert len(datasets) >= 5


def test_get_dataset_spec_known_name() -> None:
    spec = get_dataset_spec("pusht")
    assert spec.hub_repo_id == "lerobot/pusht"
    assert spec.action_dim == 2


def test_get_dataset_spec_unknown_name_raises() -> None:
    with pytest.raises(DatasetNotFoundError):
        get_dataset_spec("does-not-exist")
