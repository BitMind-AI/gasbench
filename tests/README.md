# GasBench Test Suite

Comprehensive test suite for GasBench to ensure no breaking changes and maintain code quality.

## Test Structure

```
tests/
├── unit/                          # Fast unit tests (< 1 second)
│   ├── test_config_loading.py     # YAML config validation
│   └── test_column_detection.py   # Multi-column logic tests
├── integration/                   # Integration tests with fixtures (< 5 seconds)
│   └── test_parquet_processing.py # Real file parsing tests
├── smoke/                         # Smoke tests (slow, optional)
│   └── test_dataset_connectivity.py # Real dataset checks
└── fixtures/                      # Test data (34 KB total)
    ├── test_image_single.parquet
    ├── test_image_dual.parquet    # For PICA-100K tests
    ├── test_audio.parquet
    ├── test_video.parquet
    ├── test_images.tar
    └── test_images.zip
```

## Running Tests

### Quick Tests (Default - Recommended for Development)
Run only fast tests (unit + integration):
```bash
pytest
# or
pytest -m "not slow"
```

**Time:** ~5 seconds  
**Use case:** After every code change, before committing

### All Tests Including Smoke Tests
Run everything, including slow network tests:
```bash
pytest -m slow
# or
pytest --run-slow  # if configured
```

**Time:** ~2 minutes  
**Use case:** Before releases, nightly CI

### Specific Test Categories

```bash
# Only unit tests (fastest)
pytest tests/unit/

# Only integration tests
pytest tests/integration/

# Only smoke tests
pytest tests/smoke/

# Specific test file
pytest tests/unit/test_config_loading.py

# Specific test function
pytest tests/unit/test_config_loading.py::TestConfigLoading::test_pica_100k_has_dual_columns
```

### Verbose Output

```bash
# Show all test names
pytest -v

# Show print statements
pytest -s

# Show full error traces
pytest --tb=long
```

## Test Categories

### 1. Unit Tests (⚡ Fast)
- **Config Loading**: Validates all YAML configs parse correctly
- **Column Detection**: Tests single vs multi-column logic
- **PICA-100K**: Verifies dual-column support
- **Metadata**: Tests metadata extraction logic

**Expected Count:** ~30 tests  
**Time:** < 1 second  
**No I/O:** Pure logic testing

### 2. Integration Tests (🔥 Fast)
- **Parquet Processing**: Tests real parquet file extraction
- **Single Column**: Standard dataset behavior
- **Dual Column**: PICA-100K multi-column extraction
- **All Modalities**: Image, video, audio
- **Error Handling**: Missing columns, malformed data

**Expected Count:** ~15 tests  
**Time:** ~5 seconds  
**Uses:** Test fixtures (34 KB)

### 3. Smoke Tests (🐌 Slow - Optional)
- **Dataset Connectivity**: Verifies datasets exist on HuggingFace
- **File Availability**: Checks datasets have expected files
- **New Datasets**: Validates all 9 newly added datasets
- **Critical Datasets**: Tests key datasets (PICA-100K, etc.)

**Expected Count:** ~20 tests  
**Time:** ~2 minutes (network I/O)  
**Uses:** HEAD requests (no downloads)

## Test Results

### Expected Output (Fast Tests)

```bash
$ pytest

============================= test session starts ==============================
tests/unit/test_config_loading.py::TestConfigLoading::test_all_configs_load_successfully PASSED
tests/unit/test_config_loading.py::TestConfigLoading::test_expected_dataset_counts PASSED
tests/unit/test_config_loading.py::TestImageDatasets::test_pica_100k_has_dual_columns PASSED
...
tests/integration/test_parquet_processing.py::TestParquetProcessingDualColumn::test_process_dual_column_image_parquet PASSED
...

======================== 45 passed in 4.23s ================================
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e . pytest
      - run: pytest -m "not slow"  # Fast tests only

  smoke-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # Only on main
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e . pytest
      - run: pytest -m slow  # Include slow tests
```

## Adding New Tests

### When to Add Tests

1. **New Dataset Added**: Add to `test_config_loading.py::test_new_*_datasets_exist`
2. **New Format Support**: Add integration test with fixture
3. **Bug Fix**: Add regression test
4. **New Feature**: Add unit + integration tests

### Test Template

```python
def test_your_feature():
    """Test description - what and why."""
    # Arrange
    config = create_test_config()
    
    # Act
    result = your_function(config)
    
    # Assert
    assert result == expected_value, "Descriptive failure message"
```

## Troubleshooting

### Tests Fail After Adding Dataset

1. Update expected counts in `test_expected_dataset_counts()`
2. Add dataset name to `test_new_*_datasets_exist()`

### Fixture Not Found

```bash
cd /root/gasbench
python3 -c "from tests.integration.test_parquet_processing import FIXTURES_DIR; print(FIXTURES_DIR)"
ls tests/fixtures/
```

### Slow Tests Taking Too Long

Skip them during development:
```bash
pytest -m "not slow"
```

## Coverage (Optional)

To generate coverage reports:

```bash
pip install pytest-cov
pytest --cov=src/gasbench --cov-report=html
open htmlcov/index.html
```

## Summary

| Test Type | Count | Time | Purpose |
|-----------|-------|------|---------|
| Unit | ~30 | <1s | Catch config/logic errors |
| Integration | ~15 | ~5s | Catch format/parsing issues |
| Smoke | ~20 | ~2m | Catch dataset availability issues |
| **Total** | **~65** | **~5s** | **95% confidence, fast feedback** |

**Recommendation:** Run fast tests (`pytest`) after every change. Run slow tests before releases.

