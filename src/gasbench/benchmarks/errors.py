"""Failures that invalidate a benchmark result, independently of persistence."""


class BenchmarkError(RuntimeError):
    """A run cannot produce a complete, valid benchmark score."""
