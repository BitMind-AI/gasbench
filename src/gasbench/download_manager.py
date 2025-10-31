import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import psutil

from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TaskID,
)

from .logger import get_logger
from .dataset.config import (
    BenchmarkDatasetConfig,
    discover_benchmark_image_datasets,
    discover_benchmark_video_datasets,
)
from .dataset.iterator import DatasetIterator

logger = get_logger(__name__)


@dataclass
class DatasetTask:
    """Single dataset download/processing task."""
    dataset: BenchmarkDatasetConfig
    cache_dir: str
    num_weeks: Optional[int] = None
    seed: Optional[int] = None


class DownloadManager:
    """Orchestrates concurrent dataset downloads using DatasetIterator."""
    
    def __init__(
        self,
        max_workers: int,
        cache_dir: str,
        hf_token: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.seed = seed
        self.semaphore = asyncio.Semaphore(max_workers)
        self.completed = []
        self.failed = {}
        self.active_tasks = {}  # Track active progress bars
        
    async def download_all(self, tasks: List[DatasetTask]):
        """Execute download tasks with progress tracking."""
        if not tasks:
            logger.info("No downloads needed - all datasets cached")
            return
        
        logger.info(
            f"Download plan: {len(tasks)} datasets, "
            f"{self.max_workers} concurrent workers"
        )
        
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            
            main_task = progress.add_task(
                "[cyan]Overall Progress", 
                total=len(tasks)
            )
            
            download_tasks = []
            for dataset_task in tasks:
                task = asyncio.create_task(
                    self._download_dataset_with_semaphore(
                        dataset_task, progress, main_task
                    )
                )
                download_tasks.append(task)
            
            await asyncio.gather(*download_tasks, return_exceptions=True)
        
        self._print_summary(len(tasks))
    
    async def _download_dataset_with_semaphore(
        self, 
        task: DatasetTask, 
        progress: Progress,
        main_task: TaskID
    ):
        """Download single dataset with semaphore control."""
        async with self.semaphore:
            try:
                await self._download_dataset(task, progress)
                self.completed.append(task.dataset.name)
                progress.update(main_task, advance=1)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                self.failed[task.dataset.name] = error_msg
                logger.error(f"Failed to download {task.dataset.name}: {error_msg}")
                progress.update(main_task, advance=1)
    
    async def _download_dataset(self, task: DatasetTask, progress: Progress):
        """Download and process single dataset using DatasetIterator."""
        # Create spinner for this dataset (no percentage since we can't update from sync executor)
        task_id = progress.add_task(
            f"[cyan]⏳ {task.dataset.name[:45]}", 
            total=None,  # Indeterminate spinner
            visible=True
        )
        
        try:
            loop = asyncio.get_event_loop()
            
            # Use DatasetIterator which calls download_and_extract internally
            await loop.run_in_executor(
                None,
                self._process_dataset_sync,
                task
            )
            
            # Mark as complete
            progress.update(task_id, description=f"[green]✅ {task.dataset.name[:45]}")
            
            # Give it a moment to show, then remove
            await asyncio.sleep(0.3)
            progress.remove_task(task_id)
            
        except Exception as e:
            progress.update(task_id, description=f"[red]❌ {task.dataset.name[:45]}")
            await asyncio.sleep(0.3)
            progress.remove_task(task_id)
            raise
    
    def _process_dataset_sync(self, task: DatasetTask):
        """Synchronous dataset processing using DatasetIterator."""
        import logging
        
        # Temporarily suppress INFO logs to avoid interfering with Rich progress display
        download_logger = logging.getLogger('gasbench.dataset.download')
        iterator_logger = logging.getLogger('gasbench.dataset.iterator')
        
        old_download_level = download_logger.level
        old_iterator_level = iterator_logger.level
        
        download_logger.setLevel(logging.WARNING)
        iterator_logger.setLevel(logging.WARNING)
        
        try:
            # DatasetIterator with download=True will call download_and_extract
            # Same exact same logic as gasbench run
            iterator = DatasetIterator(
                dataset_config=task.dataset,
                max_samples=2000 if "gasstation" not in task.dataset.name.lower() else 10000,
                cache_dir=task.cache_dir,
                download=True,  # This triggers ensure_cached() which calls download_and_extract
                num_weeks=task.num_weeks,
                hf_token=self.hf_token,
                seed=task.seed,
            )
        finally:
            # Restore original log levels
            download_logger.setLevel(old_download_level)
            iterator_logger.setLevel(old_iterator_level)
    
    def _print_summary(self, total: int):
        """Print download summary."""
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Completed: {len(self.completed)}/{total}")
        
        if self.failed:
            logger.info(f"❌ Failed: {len(self.failed)}/{total}")
            for name, error in list(self.failed.items())[:5]:
                logger.error(f"  {name}: {error}")
            if len(self.failed) > 5:
                logger.error(f"  ... and {len(self.failed) - 5} more failures")
        
        logger.info("=" * 60)


def get_optimal_workers() -> int:
    """Calculate optimal concurrent downloads based on system resources."""
    cpu_count = psutil.cpu_count(logical=False) or 4
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative: assume 2GB per worker for extraction + downloads
    memory_limited = max(1, int(available_memory_gb / 2))
    
    # CPU-based heuristic: 1 worker per 2 cores (I/O bound workload)
    cpu_limited = max(1, cpu_count // 2)
    
    optimal = min(memory_limited, cpu_limited, 16)
    
    logger.info(
        f"Resource calculation: {cpu_count} CPUs, {available_memory_gb:.1f}GB RAM "
        f"→ {optimal} workers (memory limit: {memory_limited}, CPU limit: {cpu_limited})"
    )
    
    return optimal


async def download_datasets(
    modality: Optional[str] = None,
    mode: str = "full",
    gasstation_only: bool = False,
    cache_dir: Optional[str] = None,
    concurrent_downloads: Optional[int] = None,
    num_weeks: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Main entry point for efficient dataset downloads.
    
    Args:
        modality: 'image', 'video', or None for all
        mode: 'debug', 'small', or 'full'
        gasstation_only: Only download gasstation datasets
        cache_dir: Cache directory path
        concurrent_downloads: Number of concurrent downloads (auto if None)
        num_weeks: Number of recent weeks for gasstation datasets
        seed: Random seed for non-gasstation dataset sampling (for reproducible random sampling)
    """
    if not cache_dir:
        cache_dir = "/.cache/gasbench"
    
    logger.info(f"Discovering datasets (modality={modality or 'all'}, mode={mode})")
    
    datasets = _discover_datasets(modality, mode, gasstation_only)
    
    if not datasets:
        logger.warning("No datasets found matching criteria")
        return
    
    logger.info(f"Found {len(datasets)} datasets to process")
    
    # Create tasks for datasets that need downloading
    tasks = []
    for dataset in datasets:
        if _needs_download(dataset, cache_dir, num_weeks):
            task = DatasetTask(
                dataset=dataset,
                cache_dir=cache_dir,
                num_weeks=num_weeks,
                seed=seed,
            )
            tasks.append(task)
        else:
            logger.info(f"Skipping {dataset.name} (already cached)")
    
    if not tasks:
        logger.info("✅ All datasets already cached")
        return
    
    workers = concurrent_downloads or get_optimal_workers()
    hf_token = os.environ.get("HF_TOKEN")
    
    manager = DownloadManager(
        max_workers=workers,
        cache_dir=cache_dir,
        hf_token=hf_token,
        seed=seed,
    )
    
    await manager.download_all(tasks)


def _discover_datasets(
    modality: Optional[str],
    mode: str,
    gasstation_only: bool,
) -> List[BenchmarkDatasetConfig]:
    """Discover datasets based on criteria."""
    datasets = []
    
    if not modality or modality == "all" or modality == "image":
        image_datasets = discover_benchmark_image_datasets(mode, gasstation_only)
        datasets.extend(image_datasets)
    
    if not modality or modality == "all" or modality == "video":
        video_datasets = discover_benchmark_video_datasets(mode, gasstation_only)
        datasets.extend(video_datasets)
    
    return datasets


def _needs_download(
    dataset: BenchmarkDatasetConfig,
    cache_dir: str,
    num_weeks: Optional[int],
) -> bool:
    """Check if dataset needs to be downloaded based on mode requirements."""
    from .dataset import gasstation_utils
    
    is_gasstation = "gasstation" in dataset.name.lower()
    
    if is_gasstation:
        target_weeks = gasstation_utils.calculate_target_weeks(num_weeks)
        for week_str in target_weeks:
            week_dir = Path(cache_dir) / "datasets" / dataset.name / week_str
            if not _is_week_cached(week_dir):
                return True
        return False
    else:
        dataset_dir = Path(cache_dir) / "datasets" / dataset.name
        return not _is_dataset_cached_for_mode(dataset_dir, dataset)


def _is_week_cached(week_dir: Path) -> bool:
    """Check if gasstation week is already cached."""
    if not week_dir.exists():
        return False
    
    samples_dir = week_dir / "samples"
    if not samples_dir.exists():
        return False
    
    samples = list(samples_dir.glob("img_*.png")) + list(samples_dir.glob("img_*.jpg"))
    return len(samples) > 10


def _get_required_samples_for_mode(dataset: BenchmarkDatasetConfig) -> int:
    """Calculate required samples based on dataset config (reflects mode)."""
    # The dataset config already has media_per_archive and archives_per_dataset
    # set by apply_mode_to_datasets() based on debug/small/full
    if dataset.media_per_archive == -1 or dataset.archives_per_dataset == -1:
        return 10000  # "full" download within reason
    
    # Calculate expected samples
    expected = dataset.media_per_archive * dataset.archives_per_dataset
    return max(expected, 10)  # At least 10 samples


def _is_dataset_cached_for_mode(dataset_dir: Path, dataset: BenchmarkDatasetConfig) -> bool:
    """Check if dataset is cached with enough samples for the requested mode."""
    if not dataset_dir.exists():
        return False
    
    metadata_file = dataset_dir / "sample_metadata.json"
    
    if not metadata_file.exists():
        return False
    
    try:
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            cached_count = len(metadata)
            required_samples = _get_required_samples_for_mode(dataset)
            
            # Check if we have enough samples for this mode
            # Allow 10% margin (e.g., 90 samples is enough for 100 required)
            return cached_count >= (required_samples * 0.9)
    except Exception:
        return False
