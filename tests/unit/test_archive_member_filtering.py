import io
import tarfile
from zipfile import ZipFile

import pytest

from gasbench.dataset.config import BenchmarkDatasetConfig
from gasbench.dataset.download.extract import yield_media_from_source


def _write_archive(path, members):
    if path.suffix == ".zip":
        with ZipFile(path, "w") as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize("format", ["tar", "zip"])
def test_archive_member_filtering_keeps_only_matching_media(tmp_path, format):
    archive_path = tmp_path / f"mixed.{format}"
    _write_archive(
        archive_path,
        {
            "raw/fake/tts/audio/fake.wav": b"synthetic audio",
            "raw/real/speech/audio/real.wav": b"captured audio",
            "raw/fake/tts/metadata/fake.json": b"{}",
            "raw/fake/excluded/other.wav": b"excluded audio",
        },
    )
    dataset = BenchmarkDatasetConfig(
        name="mixed-audio",
        path="example/mixed-audio",
        modality="audio",
        media_type="synthetic",
        source_format=format,
        archive_include_paths=["/fake/"],
        archive_exclude_paths=["/excluded/"],
    )

    samples = list(yield_media_from_source(archive_path, dataset, num_items=-1))

    assert [sample["member_path"] for sample in samples] == [
        "raw/fake/tts/audio/fake.wav"
    ]
    assert samples[0]["audio_bytes"] == b"synthetic audio"
