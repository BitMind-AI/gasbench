"""Exercise extraction using small parquet files with distinguishable payloads."""

from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import soundfile as sf
import yaml

from gasbench.dataset.config import BenchmarkDatasetConfig, load_datasets_from_yaml
from gasbench.dataset.download import _process_parquet
from gasbench.dataset.download import core


def png(red):
    stream = BytesIO()
    Image.new("RGB", (3, 2), (red, 0, 0)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def parquet(tmp_path):
    def write(rows):
        path = tmp_path / "samples.parquet"
        pd.DataFrame(rows).to_parquet(path)
        return path
    return write


@pytest.mark.parametrize("columns", [["image"], ["source_image", "edited_image"]])
@pytest.mark.parametrize("limit", [-1, 2])
def test_image_rows_preserve_pixels_metadata_and_column_pairing(parquet, columns, limit):
    rows = [
        {"row_id": row, "caption": f"caption-{row}",
         **{column: png(20 * row + col) for col, column in enumerate(columns)}}
        for row in range(4)
    ]
    path = parquet(rows)
    config = BenchmarkDatasetConfig(
        "images", "test/images", "image", "synthetic",
        data_columns=columns if len(columns) > 1 else None,
    )
    samples = list(_process_parquet(path, config, num_items=limit, seed=7))
    row_count = len(rows) if limit == -1 else limit
    assert len(samples) == row_count * len(columns)
    assert len({sample["row_id"] for sample in samples}) == row_count
    for start in range(0, len(samples), len(columns)):
        group = samples[start:start + len(columns)]
        assert len({sample["row_id"] for sample in group}) == 1
        for col, sample in enumerate(group):
            row = sample["row_id"]
            assert sample["image"].getpixel((0, 0)) == (20 * row + col, 0, 0)
            assert sample["caption"] == f"caption-{row}"
            assert sample["dataset_name"] == config.name
            assert sample["media_type"] == config.media_type
            if len(columns) > 1:
                assert sample["source_column"] == columns[col]
            else:
                assert "source_column" not in sample
    if limit != -1:
        repeat = list(_process_parquet(path, config, num_items=limit, seed=7))
        assert [s["row_id"] for s in repeat] == [s["row_id"] for s in samples]


@pytest.mark.parametrize("modality", ["audio", "video"])
def test_embedded_media_bytes_and_metadata_survive_extraction(parquet, modality):
    rows = [{modality: bytes([i, i + 1]), "description": f"sample-{i}"} for i in range(3)]
    config = BenchmarkDatasetConfig("media", "test/media", modality, "real")
    samples = list(_process_parquet(parquet(rows), config, num_items=-1))
    assert [(s[f"{modality}_bytes"], s["description"]) for s in samples] == [
        (row[modality], row["description"]) for row in rows
    ]


@pytest.mark.parametrize("columns,expected", [(["missing"], 0), (["image", "missing"], 2)])
def test_missing_requested_columns_do_not_discard_available_media(parquet, columns, expected):
    config = BenchmarkDatasetConfig("images", "test/images", "image", "real", data_columns=columns)
    path = parquet([{"image": png(10)}, {"image": png(20)}])
    samples = list(_process_parquet(path, config, num_items=-1))
    assert len(samples) == expected
    assert [s["image"].getpixel((0, 0))[0] for s in samples] == [10, 20][:expected]


def test_corrupt_media_does_not_discard_other_rows(parquet):
    path = parquet([{"image": png(10)}, {"image": b"broken"}, {"image": png(20)}])
    config = BenchmarkDatasetConfig("images", "test/images", "image", "real")
    samples = list(_process_parquet(path, config, num_items=-1))
    assert [s["image"].getpixel((0, 0))[0] for s in samples] == [10, 20]


@pytest.mark.parametrize("keep,drop", [("keep", "drop"), (0, 1), (False, True), ("", "drop")])
def test_row_filter_applies_before_sampling(parquet, keep, drop):
    path = parquet([{"image": png(i), "split": keep if i == 5 else drop} for i in range(8)])
    config = BenchmarkDatasetConfig(
        "filtered", "test/images", "image", "real", filter_column="split", filter_value=keep,
    )
    samples = list(_process_parquet(path, config, num_items=1, seed=1))
    assert len(samples) == 1
    assert samples[0]["image"].getpixel((0, 0)) == (5, 0, 0)


@pytest.mark.parametrize("label,media_type", [(0, "synthetic"), (1, "real")])
def test_numeric_yaml_filter_collects_only_matching_audio_until_target(
    tmp_path, monkeypatch, label, media_type,
):
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(yaml.safe_dump({"datasets": [{
        "name": "filtered-audio", "path": "test/audio", "modality": "audio",
        "media_type": media_type, "source_format": "parquet",
        "filter_column": "label", "filter_value": label,
    }]}))
    (config,) = load_datasets_from_yaml(str(config_path))["audio"]
    filenames = [f"shard-{i}.parquet" for i in range(4)]
    monkeypatch.setattr(core, "_list_remote_dataset_files", lambda *a, **k: filenames)
    monkeypatch.setattr(core, "_get_download_urls", lambda path, files, *a: list(files))
    downloaded = []

    def download(paths, root, **kwargs):
        for filename in paths:
            path = root / filename
            pd.DataFrame([
                {"audio": {"bytes": f"{filename}:{value}".encode()}, "label": value}
                for value in (0, 1)
            ]).to_parquet(path)
            downloaded.append(filename)
            yield path

    monkeypatch.setattr(core, "_stream_downloads", download)
    monkeypatch.setattr(core, "download_files", lambda *a, **k: list(download(*a, **k)))
    samples = list(core.download_and_extract(
        config, media_per_archive=3, archives_per_dataset=1,
        temp_dir=str(tmp_path), force_download=True, seed=7,
    ))

    # One matching row per shard requires continuing past the archive count,
    # then stopping before the fourth shard once the filtered target is met.
    assert len(samples) == len(downloaded) == 3
    assert len(set(downloaded)) == 3
    assert [sample["audio_bytes"] for sample in samples] == [
        f"{filename}:{label}".encode() for filename in downloaded
    ]
    assert all(sample["label"] == label and sample["media_type"] == media_type for sample in samples)


@pytest.mark.parametrize("metadata", [{}, {"label": 1}])
def test_zero_filter_yields_nothing_when_label_is_missing_or_unmatched(parquet, metadata):
    config = BenchmarkDatasetConfig(
        "filtered", "test/audio", "audio", "synthetic",
        filter_column="label", filter_value=0,
    )
    path = parquet([{"audio": b"excluded", **metadata}])
    assert list(_process_parquet(path, config, num_items=-1)) == []


def test_audio_array_preserves_sample_rate_and_waveform(parquet):
    waveform = np.array([0.0, 0.25, -0.5, 0.75], dtype=np.float32)
    path = parquet([{"audio": {"array": waveform, "sampling_rate": 22050}}])
    config = BenchmarkDatasetConfig("audio", "test/audio", "audio", "real")
    samples = list(_process_parquet(path, config, num_items=-1))
    assert len(samples) == 1
    decoded, rate = sf.read(BytesIO(samples[0]["audio_bytes"]))
    assert rate == 22050
    np.testing.assert_allclose(decoded, waveform, atol=1 / 32768)


@pytest.mark.parametrize("limit", [-1, 1])
def test_frame_rows_group_by_video_and_sort_by_frame_index(parquet, limit):
    path = parquet([
        {"video_id": video, "frame_idx": frame, "image": png(10 * video + frame)}
        for frame in (2, 0, 1) for video in (1, 2)
    ])
    config = BenchmarkDatasetConfig("frames", "test/frames", "video", "real")
    samples = list(_process_parquet(path, config, num_items=limit, seed=7))
    assert len(samples) == (2 if limit == -1 else limit)
    assert len({s["source_file"] for s in samples}) == len(samples)
    for sample in samples:
        video = int(sample["source_file"])
        assert sample["video_frames"] == [png(10 * video + frame) for frame in range(3)]
