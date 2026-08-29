"""Every splitter's name is a claim. These are the checks on those claims."""
import numpy as np
import pytest

from cv_traps.features.build import xy
from cv_traps.splitters import schemes


def _splits(name, df, cfg):
    _, y = xy(df, core_only=True)
    return list(schemes.build(name)(df, y, cfg.n_folds, cfg.seed))


def test_group_kfold_never_shares_a_customer(split, cfg):
    dev = split["dev"]
    for train, test in _splits("group_kfold", dev, cfg):
        shared = set(dev.iloc[train]["customer_id"]) & \
            set(dev.iloc[test]["customer_id"])
        assert not shared, f"{len(shared)} customers leaked across a fold"


def test_stratified_kfold_shares_almost_every_customer(split, cfg):
    """The trap has to actually be present, or the comparison is empty."""
    dev = split["dev"]
    report = schemes.leakage_report(dev, _splits("stratified_kfold", dev, cfg))
    assert report["pct_test_rows_seen_customer"].min() > 0.80


def test_forward_chaining_never_trains_on_the_future(split, cfg):
    dev = split["dev"]
    for train, test in _splits("forward_chaining", dev, cfg):
        assert dev.iloc[train]["period"].max() < dev.iloc[test]["period"].min()


def test_grouped_forward_chaining_respects_both_constraints(split, cfg):
    dev = split["dev"]
    for train, test in _splits("grouped_forward_chaining", dev, cfg):
        assert dev.iloc[train]["period"].max() < dev.iloc[test]["period"].min()
        assert not (set(dev.iloc[train]["customer_id"])
                    & set(dev.iloc[test]["customer_id"]))


def test_group_kfold_ignores_the_seed_but_shuffled_group_does_not(split, cfg):
    """The bug that made the variance experiment report a spread of 0.0000.

    GroupKFold is deterministic, so twenty 'repeats' built on it are the same
    split twenty times. This pins both halves of that: the plain one really is
    seed-blind, and the shuffled one really does respond.
    """
    dev = split["dev"]
    _, y = xy(dev, core_only=True)
    a = [t.tolist() for _, t in schemes.grouped(dev, y, cfg.n_folds, 1)]
    b = [t.tolist() for _, t in schemes.grouped(dev, y, cfg.n_folds, 999)]
    assert a == b, "GroupKFold is documented as seed-blind but moved"

    c = [t.tolist() for _, t in schemes.shuffled_grouped(dev, y, cfg.n_folds, 1)]
    d = [t.tolist() for _, t in
         schemes.shuffled_grouped(dev, y, cfg.n_folds, 999)]
    assert c != d, "shuffled_grouped must respond to the seed"


def test_every_row_is_tested_exactly_once_by_kfold(split, cfg):
    dev = split["dev"]
    seen = np.concatenate([t for _, t in _splits("group_kfold", dev, cfg)])
    assert len(seen) == len(dev)
    assert len(np.unique(seen)) == len(dev)


def test_unknown_scheme_fails_loudly():
    with pytest.raises(KeyError, match="unknown scheme"):
        schemes.build("kfold_but_better")
