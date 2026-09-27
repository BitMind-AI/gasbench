from dataclasses import replace
from types import SimpleNamespace

import pytest

from gasbench.benchmarks import common


@pytest.mark.parametrize("pattern", ["SELECTED", "gasstation", "no-match"])
def test_filtered_run_retains_full_run_sample_limits(monkeypatch, pattern):
    datasets = [
        SimpleNamespace(name=name, media_type="real")
        for name in ("selected", "other", "gasstation-samples")
    ]
    monkeypatch.setattr(common, "discover_benchmark_datasets", lambda **kwargs: datasets)
    monkeypatch.setattr(common, "get_benchmark_size", lambda *args: 12)
    monkeypatch.setattr(common, "build_dataset_info", lambda *args: {})
    config = common.BenchmarkRunConfig(
        modality="video", mode="full", gasstation_only=False,
        dataset_config_path=None, holdout_config_path=None, cache_dir="unused",
        hf_token=None, batch_size=1, augment_level=0, crop_prob=0,
        records_parquet_path=None,
    )
    logger = SimpleNamespace(info=lambda *args: None)
    full = common.build_plan(logger, config, [])
    filtered = common.build_plan(logger, replace(config, dataset_filters=[pattern]), [])
    expected = {name: cap for name, cap in full.sampling_plan.items() if pattern.lower() in name}
    if not expected:
        assert filtered is None
    else:
        assert filtered.sampling_plan == expected
        assert [d.name for d in filtered.available_datasets] == list(expected)
