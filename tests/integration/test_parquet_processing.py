"""
Integration tests for parquet file processing.

These tests use real parquet fixtures to verify extraction logic.
"""

import pytest
from pathlib import Path
from src.gasbench.dataset.config import BenchmarkDatasetConfig
from src.gasbench.dataset.download import _process_parquet


# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestParquetProcessingSingleColumn:
    """Test parquet processing with single media column."""

    def test_process_single_column_image_parquet(self):
        """Test processing parquet with single image column."""
        fixture_path = FIXTURES_DIR / "test_image_single.parquet"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
        
        # Create test config
        config = BenchmarkDatasetConfig(
            name="test_single",
            path="test/test",
            modality="image",
            media_type="real",
        )
        
        # Process parquet
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # Verify results
        assert len(samples) == 5, "Should yield 5 samples (1 per row)"
        
        # Check first sample
        # Note: _create_sample returns PIL Image object, not bytes
        assert "image" in samples[0] or "image_bytes" in samples[0]
        assert "dataset_name" in samples[0]
        assert "media_type" in samples[0]
        assert samples[0]["dataset_name"] == "test_single"
        
        # Should NOT have source_column for single-column dataset
        assert "source_column" not in samples[0]

    def test_single_column_includes_metadata(self):
        """Test that parquet row metadata is included."""
        fixture_path = FIXTURES_DIR / "test_image_single.parquet"
        
        config = BenchmarkDatasetConfig(
            name="test_single",
            path="test/test",
            modality="image",
            media_type="real",
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # Check that metadata columns are included
        assert "caption" in samples[0]
        assert "id" in samples[0]
        assert samples[0]["caption"] == "test caption"

    def test_single_column_respects_num_items(self):
        """Test that num_items parameter works."""
        fixture_path = FIXTURES_DIR / "test_image_single.parquet"
        
        config = BenchmarkDatasetConfig(
            name="test_single",
            path="test/test",
            modality="image",
            media_type="real",
        )
        
        # Request only 2 samples
        samples = list(_process_parquet(fixture_path, config, num_items=2))
        
        assert len(samples) == 2, "Should yield only 2 samples"


class TestParquetProcessingDualColumn:
    """Test parquet processing with multiple media columns (PICA-100K style)."""

    def test_process_dual_column_image_parquet(self):
        """Test processing parquet with dual image columns."""
        fixture_path = FIXTURES_DIR / "test_image_dual.parquet"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
        
        config = BenchmarkDatasetConfig(
            name="test_dual",
            path="test/test",
            modality="image",
            media_type="synthetic",
            data_columns=["src_img", "tgt_img"],
        )
        
        # Process parquet
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # Verify results: 5 rows × 2 columns = 10 samples
        assert len(samples) == 10, f"Should yield 10 samples (5 rows × 2 columns), got {len(samples)}"

    def test_dual_column_has_source_column(self):
        """Test that dual-column datasets add source_column metadata."""
        fixture_path = FIXTURES_DIR / "test_image_dual.parquet"
        
        config = BenchmarkDatasetConfig(
            name="test_dual",
            path="test/test",
            modality="image",
            media_type="synthetic",
            data_columns=["src_img", "tgt_img"],
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # Check source_column is set
        assert "source_column" in samples[0]
        assert "source_column" in samples[1]
        
        # First sample should be from src_img, second from tgt_img
        assert samples[0]["source_column"] == "src_img"
        assert samples[1]["source_column"] == "tgt_img"

    def test_dual_column_alternates_correctly(self):
        """Test that samples alternate between columns correctly."""
        fixture_path = FIXTURES_DIR / "test_image_dual.parquet"
        
        config = BenchmarkDatasetConfig(
            name="test_dual",
            path="test/test",
            modality="image",
            media_type="synthetic",
            data_columns=["src_img", "tgt_img"],
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # Check pattern: src_img, tgt_img, src_img, tgt_img, ...
        for i, sample in enumerate(samples):
            expected_col = "src_img" if i % 2 == 0 else "tgt_img"
            assert sample["source_column"] == expected_col, \
                f"Sample {i} should be from {expected_col}, got {sample['source_column']}"

    def test_dual_column_includes_shared_metadata(self):
        """Test that both samples from same row have same metadata."""
        fixture_path = FIXTURES_DIR / "test_image_dual.parquet"
        
        config = BenchmarkDatasetConfig(
            name="test_dual",
            path="test/test",
            modality="image",
            media_type="synthetic",
            data_columns=["src_img", "tgt_img"],
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # First two samples are from same row
        sample_0 = samples[0]
        sample_1 = samples[1]
        
        # They should have same metadata
        assert sample_0["caption"] == sample_1["caption"]
        assert sample_0["video_id"] == sample_1["video_id"]

    def test_dual_column_respects_num_items(self):
        """Test that num_items affects row sampling, not final sample count."""
        fixture_path = FIXTURES_DIR / "test_image_dual.parquet"
        
        config = BenchmarkDatasetConfig(
            name="test_dual",
            path="test/test",
            modality="image",
            media_type="synthetic",
            data_columns=["src_img", "tgt_img"],
        )
        
        # Request 2 rows: should get 4 samples (2 rows × 2 columns)
        samples = list(_process_parquet(fixture_path, config, num_items=2))
        
        assert len(samples) == 4, \
            f"Should yield 4 samples (2 rows × 2 columns), got {len(samples)}"


class TestParquetProcessingAudio:
    """Test parquet processing for audio modality."""

    def test_process_audio_parquet(self):
        """Test processing audio parquet file."""
        fixture_path = FIXTURES_DIR / "test_audio.parquet"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
        
        config = BenchmarkDatasetConfig(
            name="test_audio",
            path="test/test",
            modality="audio",
            media_type="real",
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        assert len(samples) == 5, "Should yield 5 audio samples"
        assert "audio_bytes" in samples[0]
        assert "transcript" in samples[0]


class TestParquetProcessingVideo:
    """Test parquet processing for video modality."""

    def test_process_video_parquet(self):
        """Test processing video parquet file."""
        fixture_path = FIXTURES_DIR / "test_video.parquet"
        assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
        
        config = BenchmarkDatasetConfig(
            name="test_video",
            path="test/test",
            modality="video",
            media_type="real",
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        assert len(samples) == 3, "Should yield 3 video samples"
        assert "video_bytes" in samples[0]
        assert "description" in samples[0]


class TestParquetErrorHandling:
    """Test error handling in parquet processing."""

    def test_missing_column_returns_empty(self):
        """Test that missing media columns are handled gracefully."""
        fixture_path = FIXTURES_DIR / "test_image_single.parquet"
        
        # Config specifies columns that don't exist
        config = BenchmarkDatasetConfig(
            name="test_missing",
            path="test/test",
            modality="image",
            media_type="real",
            data_columns=["nonexistent_col"],
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # Should return empty (warning logged)
        assert len(samples) == 0, "Should return no samples when columns don't exist"

    def test_partial_missing_columns_uses_available(self):
        """Test that if some columns exist, they are used."""
        fixture_path = FIXTURES_DIR / "test_image_dual.parquet"
        
        # Config specifies one existing, one non-existing
        config = BenchmarkDatasetConfig(
            name="test_partial",
            path="test/test",
            modality="image",
            media_type="synthetic",
            data_columns=["src_img", "nonexistent"],
        )
        
        samples = list(_process_parquet(fixture_path, config, num_items=-1))
        
        # Should use only src_img: 5 rows × 1 column = 5 samples
        assert len(samples) == 5, f"Should yield 5 samples (only src_img), got {len(samples)}"
        # Note: source_column only added when len(media_cols) > 1, so it won't be there if only 1 column found
        # assert all(s.get("source_column") == "src_img" for s in samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

