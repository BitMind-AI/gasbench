import os
import contextlib
import shutil
import time
import tarfile
from typing import Dict, Optional, Iterator

from huggingface_hub import hf_hub_download

from ..logger import get_logger

logger = get_logger(__name__)


@contextlib.contextmanager
def video_archive_manager(cache_dir: str = "/.cache/gasbench") -> Iterator["VideoArchiveCache"]:
    """Context manager for efficient video archive extraction and caching."""
    cache = VideoArchiveCache(cache_dir=cache_dir)
    try:
        yield cache
    finally:
        pass


class VideoArchiveCache:
    """Simple caching and extraction of video archives."""

    def __init__(self, cache_dir: str = "/.cache/gasbench"):
        self.volume_dir = cache_dir
        self.archives_dir = os.path.join(self.volume_dir, "video_archives")
        os.makedirs(self.archives_dir, exist_ok=True)

    def get_video_path(self, sample: Dict) -> Optional[str]:
        """Get the local path to a video, unpacking archive if needed."""
        hf_cache_dir = os.path.join(self.volume_dir, "temp_downloads", "hf_cache")

        try:
            archive_filename = sample.get("archive_filename")
            video_path_in_archive = sample.get("video_path_in_archive")

            if not archive_filename or not video_path_in_archive:
                logger.warning(
                    f"Missing archive info: archive={archive_filename}, video_path={video_path_in_archive}"
                )
                return None

            archive_name = archive_filename.replace(".tar.gz", "").replace(".tar", "")
            cache_dir = os.path.join(self.archives_dir, archive_name)
            video_path = os.path.join(cache_dir, video_path_in_archive)

            if os.path.exists(cache_dir) and os.path.exists(video_path):
                logger.debug(f"📁 Using cached video: {video_path}")
                return video_path
            elif os.path.exists(cache_dir):
                completion_marker = os.path.join(cache_dir, ".extraction_complete")
                if os.path.exists(completion_marker):
                    logger.warning(f"❌ Video not found in completed archive: {video_path}")
                    logger.debug("Archive was marked complete but video missing")

                    if not hasattr(self, "_logged_structure_for"):
                        self._logged_structure_for = set()

                    if cache_dir not in self._logged_structure_for:
                        try:
                            contents = os.listdir(cache_dir)
                            logger.debug(f"📁 Completed archive contents: {contents}")
                            self._logged_structure_for.add(cache_dir)
                        except:
                            pass

                    return None
                else:
                    logger.debug("Archive directory exists but extraction incomplete, will re-extract")
                    shutil.rmtree(cache_dir)

            logger.info(f"📥 Downloading archive: {archive_filename}")
            return self._download_and_extract_archive(
                archive_filename, cache_dir, video_path, hf_cache_dir
            )

        except Exception as e:
            logger.error(f"Failed to extract video from archive: {e}")
            return None

    def _download_and_extract_archive(
        self, archive_filename: str, cache_dir: str, video_path: str, hf_cache_dir: str
    ) -> Optional[str]:
        """Download and extract archive with error handling."""
        logger.info(f"📦 Downloading archive: {archive_filename}")

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    possible_paths = [
                        f"archives/2025W38/{archive_filename}",
                        f"archives/{archive_filename}",
                        archive_filename,
                    ]

                    tar_path = None
                    last_error = None

                    for archive_path in possible_paths:
                        try:
                            logger.debug(f"🔍 Trying archive path: {archive_path}")
                            tar_path = hf_hub_download(
                                repo_id="gasstation/generated-videos",
                                filename=archive_path,
                                repo_type="dataset",
                                cache_dir=hf_cache_dir,
                            )
                            logger.info(f"✅ Downloaded archive from: {archive_path}")
                            break
                        except Exception as path_error:
                            last_error = path_error
                            error_msg = str(path_error)
                            if "404" in error_msg or "Entry Not Found" in error_msg:
                                logger.debug(f"📭 Archive not found at: {archive_path}")
                                continue
                            else:
                                raise path_error

                    if tar_path:
                        break
                    else:
                        raise last_error

                except Exception as download_error:
                    error_msg = str(download_error)
                    if "404" in error_msg or "Entry Not Found" in error_msg:
                        logger.warning(f"📭 Archive not available on HuggingFace: {archive_filename}")
                        logger.debug(f"This is expected if archives haven't been uploaded yet: {error_msg}")
                        return None
                    elif "401" in error_msg or "403" in error_msg:
                        logger.warning(f"🔒 Access denied for archive: {archive_filename}")
                        return None
                    elif attempt < max_retries - 1:
                        logger.warning(f"⚠️ Download attempt {attempt + 1} failed: {download_error}")
                        time.sleep(2**attempt)
                        continue
                    else:
                        logger.error(f"❌ All download attempts failed: {download_error}")
                        return None
            else:
                logger.error(f"❌ Failed to download after {max_retries} attempts")
                return None

            logger.info(f"📂 Extracting archive to: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)

            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(cache_dir)

                completion_marker = os.path.join(cache_dir, ".extraction_complete")
                with open(completion_marker, "w") as f:
                    f.write(f"Extracted at {time.time()}")

                logger.info(f"✅ Archive extracted successfully")

            except Exception as extract_error:
                logger.error(f"❌ Failed to extract archive: {extract_error}")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                return None

            if os.path.exists(video_path):
                logger.info(f"✅ Video ready: {video_path}")
                return video_path
            else:
                logger.error(f"❌ Video not found at expected path: {video_path}")

                video_filename = os.path.basename(video_path)
                logger.debug(f"🔍 Searching for video file: {video_filename}")

                found_video = None
                try:
                    for root, dirs, files in os.walk(cache_dir):
                        if video_filename in files:
                            found_video = os.path.join(root, video_filename)
                            logger.info(
                                f"🎯 Found video at alternative location: {found_video}"
                            )
                            break

                    if found_video:
                        return found_video

                except Exception as e:
                    logger.warning(f"Failed to search for video file: {e}")

                # List contents for debugging (only if we haven't found the video)
                try:

                    def list_directory_tree(directory, prefix=""):
                        items = []
                        try:
                            for item in os.listdir(directory):
                                item_path = os.path.join(directory, item)
                                if os.path.isdir(item_path):
                                    items.append(f"{prefix}{item}/")
                                    items.extend(
                                        list_directory_tree(item_path, prefix + "  ")
                                    )
                                else:
                                    items.append(f"{prefix}{item}")
                        except:
                            pass
                        return items

                    tree = list_directory_tree(cache_dir)
                    logger.debug(f"📁 Archive directory structure:")
                    for item in tree[:20]:  # Limit to first 20 items
                        logger.debug(f"  {item}")
                    if len(tree) > 20:
                        logger.debug(f"  ... and {len(tree) - 20} more items")

                except Exception as e:
                    logger.warning(f"Failed to list archive contents: {e}")

                return None

        except Exception as e:
            logger.error(f"Failed to download and extract archive: {e}")
            return None

    def get_cache_info(self) -> Dict[str, any]:
        """Get information about cached archives."""
        cached_count = 0
        if os.path.exists(self.archives_dir):
            cached_count = len(
                [
                    d
                    for d in os.listdir(self.archives_dir)
                    if os.path.isdir(os.path.join(self.archives_dir, d))
                ]
            )

        return {
            "volume_dir": self.volume_dir,
            "archives_dir": self.archives_dir,
            "unpacked_archives": cached_count,
        }
