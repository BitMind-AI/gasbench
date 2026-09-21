"""Current-week gasstation fallback must use the local cache when HF is down."""
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

from gasbench.dataset.utils import gasstation_utils


def _iso_week(offset_weeks: int = 0) -> str:
    date = datetime.now() - timedelta(weeks=offset_weeks)
    year, week, _ = date.isocalendar()
    return f"{year}W{week:02d}"


def _write_week(cache_dir: Path, dataset_name: str, week: str, n: int) -> None:
    week_dir = cache_dir / "datasets" / dataset_name / week
    week_dir.mkdir(parents=True)
    metadata = {f"sample_{i}.jpg": {} for i in range(n)}
    (week_dir / "sample_metadata.json").write_text(json.dumps(metadata))


def _block_hf(monkeypatch):
    fake = types.ModuleType("gasbench.dataset.download")
    fake.list_hf_files = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("blocked"))
    monkeypatch.setitem(sys.modules, "gasbench.dataset.download", fake)


def test_fallback_uses_cached_previous_week_when_hf_fails(tmp_path, monkeypatch):
    dataset = "gasstation-generated-images"
    _write_week(tmp_path, dataset, _iso_week(1), 20)
    _write_week(tmp_path, dataset, _iso_week(0), 0)
    _block_hf(monkeypatch)

    weeks = gasstation_utils.calculate_target_weeks(
        dataset_path="gasstation/gs-images-v4",
        source_format="parquet",
        modality="image",
        cache_dir=str(tmp_path),
        dataset_name=dataset,
    )
    assert _iso_week(0) in weeks
    assert _iso_week(1) in weeks


def test_populated_current_week_is_kept_without_hf(tmp_path, monkeypatch):
    dataset = "gasstation-generated-images"
    _write_week(tmp_path, dataset, _iso_week(0), 8)
    _write_week(tmp_path, dataset, _iso_week(1), 20)
    _block_hf(monkeypatch)

    weeks = gasstation_utils.calculate_target_weeks(
        dataset_path="gasstation/gs-images-v4",
        source_format="parquet",
        modality="image",
        cache_dir=str(tmp_path),
        dataset_name=dataset,
    )
    assert weeks == [_iso_week(0)]


def test_hf_failure_without_cache_still_assumes_current_week(monkeypatch):
    _block_hf(monkeypatch)

    weeks = gasstation_utils.calculate_target_weeks(
        dataset_path="gasstation/gs-images-v4",
        source_format="parquet",
        modality="image",
    )
    assert weeks == [_iso_week(0)]
