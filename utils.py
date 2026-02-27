"""
Utility Functions for Affective Computing Pipeline
Common operations and helper functions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration management for the pipeline."""
    
    DEFAULT_CONFIG = {
        "frame_extraction": {
            "interval_seconds": 1.0,
            "output_format": "jpg",
            "quality": 85
        },
        "embedding": {
            "model": "openai/clip-vit-base-patch32",
            "device": "auto",
            "batch_size": 32
        },
        "similarity": {
            "metric": "cosine",
            "top_k": 5,
            "heatmap_cmap": "YlOrRd"
        }
    }
    
    def __init__(self, config_path: str = None):
        """Load or create configuration."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._merge_config(user_config)
            logger.info(f"Loaded config from {config_path}")
    
    def _merge_config(self, user_config: Dict):
        """Recursively merge user config with defaults."""
        for key, value in user_config.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def save(self, output_path: str):
        """Save configuration to file."""
        Path(output_path).parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Config saved to {output_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation (e.g., 'embedding.model')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


class LogManager:
    """Centralized logging management."""
    
    @staticmethod
    def setup_logging(
        log_file: str = None,
        level: str = "INFO"
    ):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        handlers = [
            logging.StreamHandler()
        ]
        
        if log_file:
            Path(log_file).parent.mkdir(exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=getattr(logging, level),
            format=log_format,
            handlers=handlers
        )
        
        return logging.getLogger(__name__)


class MetadataManager:
    """Manage pipeline metadata and tracking."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "pipeline_stages": {}
        }
    
    def record_stage(
        self,
        stage_name: str,
        status: str,
        details: Dict = None,
        error: str = None
    ):
        """Record a pipeline stage execution."""
        self.metadata["pipeline_stages"][stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "status": status,  # 'success', 'failed', 'skipped'
            "details": details or {},
            "error": error
        }
    
    def save(self, output_path: str = "pipeline_metadata.json"):
        """Save metadata to file."""
        output_path = self.base_dir / output_path
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_path}")


def setup_directories(
    base_dir: str = ".",
    dirs: List[str] = None
) -> Dict[str, Path]:
    """
    Create and verify directory structure.
    
    Args:
        base_dir: Root directory
        dirs: List of subdirectories to create
        
    Returns:
        Dictionary mapping names to Path objects
    """
    if dirs is None:
        dirs = ["videos", "frames", "embeddings", "outputs"]
    
    paths = {}
    for dir_name in dirs:
        path = Path(base_dir) / dir_name
        path.mkdir(exist_ok=True)
        paths[dir_name] = path
    
    logger.info(f"Directory structure verified in {base_dir}")
    return paths


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required packages are installed.
    
    Returns:
        Dictionary of package names and availability
    """
    required_packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'transformers': 'transformers',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib'
    }
    
    availability = {}
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            availability[package_name] = True
        except ImportError:
            availability[package_name] = False
            logger.warning(f"Missing package: {package_name}")
    
    return availability


def get_system_info() -> Dict[str, Any]:
    """Get system and environment information."""
    import sys
    import platform
    import torch
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info["gpu_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info


if __name__ == "__main__":
    # Example usage
    LogManager.setup_logging("logs/pipeline.log")
    logger = logging.getLogger(__name__)
    
    logger.info("System Information:")
    for key, value in get_system_info().items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nDependency Check:")
    deps = check_dependencies()
    for package, available in deps.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {package}")
