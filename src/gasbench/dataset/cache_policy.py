import json
from pathlib import Path
from typing import Dict, Optional

from ..logger import get_logger

logger = get_logger(__name__)


def load_cache_policy(policy_path: Optional[str]) -> Optional[Dict]:
    """Load cache policy from JSON file.
    
    Args:
        policy_path: Path to policy JSON file (None means no policy)
        
    Returns:
        Policy dict with generator_priorities, or None if no policy/invalid
    """
    if not policy_path:
        return None
    
    try:
        with open(policy_path, "r") as f:
            policy = json.load(f)
        
        # Validate structure
        if "generator_priorities" not in policy:
            logger.warning(f"Invalid policy file: missing generator_priorities")
            return None
        
        num_generators = len(policy["generator_priorities"])
        logger.info(f"Loaded cache policy with {num_generators} generators")
        
        if num_generators > 0:
            avg_priority = sum(policy["generator_priorities"].values()) / num_generators
            logger.info(f"Average generator priority: {avg_priority:.3f}")
        
        return policy
        
    except FileNotFoundError:
        logger.debug(f"Cache policy file not found: {policy_path}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in cache policy file: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load cache policy: {e}")
        return None


def get_generator_priority(policy: Optional[Dict], hotkey: str) -> float:
    """Get priority for a generator from the policy.
    
    Args:
        policy: Loaded cache policy dict (or None if no policy)
        hotkey: Generator hotkey (SS58 address)
        
    Returns:
        Priority value (0.0-1.0), or 0.5 (default) if no policy or generator not in policy
    """
    if not policy:
        return 0.5
    priorities = policy.get("generator_priorities", {})
    return priorities.get(hotkey, 0.5)

