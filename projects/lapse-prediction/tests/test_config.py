"""Config validation catches business settings that would silently corrupt the
target definition."""
import dataclasses

import pytest

from lapse_prediction.config import CFG, Config, load_config


def test_default_config_is_valid():
    assert CFG.n_classes == len(CFG.buckets) + 1
    assert CFG.class_names[-1] == CFG.lapse_label
    assert CFG.edges[-1] == CFG.grace_days


def test_last_bucket_edge_must_equal_grace():
    bad = dataclasses.replace(CFG, grace_days=60)
    with pytest.raises(ValueError, match="must equal grace_days"):
        bad.validate()


def test_buckets_must_ascend():
    bad = dataclasses.replace(CFG, buckets=(("a", 30), ("b", 7), ("c", 45)))
    with pytest.raises(ValueError, match="ascending"):
        bad.validate()


def test_hazard_horizon_must_cover_grace():
    bad = dataclasses.replace(CFG, hazard_horizon_days=10)
    with pytest.raises(ValueError, match="grace"):
        bad.validate()


def test_yaml_round_trip_matches_defaults():
    cfg = load_config()
    assert cfg.grace_days == 45
    assert cfg.lapse_index == len(cfg.buckets)
