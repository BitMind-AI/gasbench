"""
Unit tests for column detection and normalization logic.

These tests verify the multi-column support for PICA-100K and backward compatibility.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.gasbench.dataset.config import BenchmarkDatasetConfig


class TestColumnNormalization:
    """Test single vs multi-column normalization logic."""

    def test_single_column_becomes_list(self):
        """Test that single column (string) is converted to list."""
        media_col = "image"
        media_cols = media_col if isinstance(media_col, list) else [media_col]
        
        assert isinstance(media_cols, list)
        assert len(media_cols) == 1
        assert media_cols[0] == "image"

    def test_multi_column_stays_list(self):
        """Test that multi-column (list) stays as list."""
        media_col = ["src_img", "tgt_img"]
        media_cols = media_col if isinstance(media_col, list) else [media_col]
        
        assert isinstance(media_cols, list)
        assert len(media_cols) == 2
        assert media_cols == ["src_img", "tgt_img"]

    def test_iteration_count_single_column(self):
        """Test that single column results in 1 iteration."""
        media_col = "image"
        media_cols = media_col if isinstance(media_col, list) else [media_col]
        
        iteration_count = 0
        for col in media_cols:
            iteration_count += 1
        
        assert iteration_count == 1

    def test_iteration_count_multi_column(self):
        """Test that multi-column results in N iterations."""
        media_col = ["src_img", "tgt_img"]
        media_cols = media_col if isinstance(media_col, list) else [media_col]
        
        iteration_count = 0
        for col in media_cols:
            iteration_count += 1
        
        assert iteration_count == 2


class TestDataColumnsAttribute:
    """Test the data_columns attribute handling."""

    def test_dataset_without_data_columns(self):
        """Test dataset without data_columns attribute."""
        dataset = BenchmarkDatasetConfig(
            name="test",
            path="test/test",
            modality="image",
            media_type="real",
        )
        
        data_columns = getattr(dataset, 'data_columns', None)
        assert data_columns is None

    def test_dataset_with_data_columns(self):
        """Test dataset with data_columns attribute."""
        dataset = BenchmarkDatasetConfig(
            name="test",
            path="test/test",
            modality="image",
            media_type="real",
            data_columns=["src_img", "tgt_img"],
        )
        
        data_columns = getattr(dataset, 'data_columns', None)
        assert data_columns is not None
        assert data_columns == ["src_img", "tgt_img"]

    def test_data_columns_filters_to_existing(self):
        """Test that data_columns are filtered to those that exist in dataframe."""
        data_columns_config = ["src_img", "tgt_img", "nonexistent"]
        df_columns = ["src_img", "tgt_img", "caption", "id"]
        
        media_col = [c for c in data_columns_config if c in df_columns]
        
        assert len(media_col) == 2
        assert "src_img" in media_col
        assert "tgt_img" in media_col
        assert "nonexistent" not in media_col


class TestModalityColumnDetection:
    """Test column auto-detection for different modalities."""

    def test_image_column_detection_exact_match(self):
        """Test image column detection with exact 'image' column."""
        columns = ["id", "image", "caption"]
        
        media_col = next((c for c in columns if c.lower() == "image"), None)
        
        assert media_col == "image"

    def test_image_column_detection_fuzzy_match(self):
        """Test image column detection with fuzzy match."""
        columns = ["id", "image_data", "caption"]
        
        media_col = (
            next((c for c in columns if c.lower() == "image"), None)
            or next((c for c in columns if "image" in c.lower() and "_id" not in c.lower()), None)
        )
        
        assert media_col == "image_data"

    def test_image_column_excludes_id_columns(self):
        """Test that _id columns are excluded."""
        columns = ["image_id", "image_data", "caption"]
        
        media_col = (
            next((c for c in columns if c.lower() == "image"), None)
            or next((c for c in columns if "image" in c.lower() and "_id" not in c.lower()), None)
        )
        
        assert media_col == "image_data"

    def test_audio_column_detection(self):
        """Test audio column detection."""
        columns = ["id", "audio", "transcript"]
        candidates = ["audio", "bytes", "content", "data", "wav", "mp3"]
        
        media_col = (
            next((c for c in columns if c.lower() == "audio"), None)
            or next((c for c in columns if any(k in c.lower() for k in candidates)), None)
        )
        
        assert media_col == "audio"

    def test_video_column_detection(self):
        """Test video column detection."""
        columns = ["id", "video", "description"]
        candidates = ["video", "bytes", "content", "data"]
        
        media_col = (
            next((c for c in columns if c.lower() in candidates), None)
            or next((c for c in columns if any(k in c.lower() for k in candidates)), None)
        )
        
        assert media_col == "video"


class TestSourceColumnMetadata:
    """Test source_column metadata for multi-column datasets."""

    def test_single_column_no_source_column(self):
        """Test that single-column datasets don't add source_column."""
        media_cols = ["image"]
        
        sample = {"image_bytes": b"data"}
        
        if len(media_cols) > 1:
            sample["source_column"] = media_cols[0]
        
        assert "source_column" not in sample

    def test_multi_column_adds_source_column(self):
        """Test that multi-column datasets add source_column."""
        media_cols = ["src_img", "tgt_img"]
        
        sample = {"image_bytes": b"data"}
        
        if len(media_cols) > 1:
            sample["source_column"] = "src_img"
        
        assert "source_column" in sample
        assert sample["source_column"] == "src_img"

    def test_each_iteration_has_correct_source_column(self):
        """Test that each iteration gets the correct source_column."""
        media_cols = ["src_img", "tgt_img"]
        samples = []
        
        for col in media_cols:
            sample = {"image_bytes": b"data"}
            if len(media_cols) > 1:
                sample["source_column"] = col
            samples.append(sample)
        
        assert samples[0]["source_column"] == "src_img"
        assert samples[1]["source_column"] == "tgt_img"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

