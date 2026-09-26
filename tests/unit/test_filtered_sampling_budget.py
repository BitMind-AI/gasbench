from types import SimpleNamespace

from gasbench.benchmarks import common


def test_filtered_run_retains_full_run_sample_limits(monkeypatch):
    datasets = [
        SimpleNamespace(name=name, media_type="real")
        for name in ("selected", "other")
    ]
    monkeypatch.setattr(common, "discover_benchmark_datasets", lambda **kwargs: datasets)
    monkeypatch.setattr(common, "get_benchmark_size", lambda *args: 12)
    monkeypatch.setattr(common, "build_dataset_info", lambda *args: {})
    config = common.BenchmarkRunConfig(
        modality="video", mode="full", gasstation_only=False,
        dataset_config_path=None, holdout_config_path=None, cache_dir="unused",
        hf_token=None, batch_size=1, augment_level=0, crop_prob=0,
        records_parquet_path=None, dataset_filters=["selected"],
    )
    logger = SimpleNamespace(info=lambda *args: None)
    plan = common.build_plan(logger, config, [])
    assert [d.name for d in plan.available_datasets] == ["selected"]
    assert plan.sampling_plan == {"selected": 6}
