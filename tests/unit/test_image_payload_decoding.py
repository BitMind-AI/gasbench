import base64
from io import BytesIO

from PIL import Image

from gasbench.dataset.config import BenchmarkDatasetConfig
from gasbench.dataset.download.extract import yield_media_from_source


def _image_config() -> BenchmarkDatasetConfig:
    return BenchmarkDatasetConfig(
        name="encoded-images",
        path="test/images",
        modality="image",
        media_type="synthetic",
        source_format="png",
    )


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), color="red").save(output, format="PNG")
    return output.getvalue()


def test_direct_image_accepts_base64_wrapped_png(tmp_path):
    source = tmp_path / "encoded.png"
    source.write_bytes(base64.b64encode(_png_bytes()))

    samples = list(yield_media_from_source(source, _image_config(), num_items=-1))

    assert len(samples) == 1
    assert samples[0]["image"].size == (3, 2)


def test_direct_image_accepts_base64_data_url(tmp_path):
    source = tmp_path / "encoded.png"
    source.write_bytes(b"data:image/png;base64," + base64.b64encode(_png_bytes()))

    samples = list(yield_media_from_source(source, _image_config(), num_items=-1))

    assert len(samples) == 1
    assert samples[0]["image"].size == (3, 2)


def test_direct_image_rejects_invalid_raw_and_base64_payload(tmp_path):
    source = tmp_path / "broken.png"
    source.write_bytes(b"not an image or valid base64!")

    assert list(yield_media_from_source(source, _image_config(), num_items=-1)) == []
