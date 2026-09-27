# Tests

Install the development dependencies from the repository root:

```bash
pip install -e '.[dev]'
```

Run the offline suite and lint checks:

```bash
pytest
ruff check src tests
```

The offline suite covers configuration and sample selection, extraction and
caching, model inference, transforms, scoring, and checkpoint recovery. External
services are replaced with test doubles; parquet and model fixtures are generated
in temporary directories with distinguishable inputs.

CI runs the full offline suite on CPU. Separate base-install checks ensure the
dataset registry and checkpoint reader work without the inference dependencies.

## Live dataset checks

```bash
pytest -m slow tests/smoke
```

These optional checks list files for one configured Hugging Face dataset per
modality. They require network access and any credentials needed by the selected
datasets. They check availability, not full downloads or model quality.

## Test quality

- Test behavior, an interface contract, an invariant, or a demonstrated failure.
- Identify a plausible defect the test would catch before adding it.
- Avoid pinning mutable registry entries, revisions, tuning values, or test counts.
- Use distinguishable inputs and assert outcomes, not just that code ran.
- Consolidate redundant cases; retain cases that exercise different failure paths.
