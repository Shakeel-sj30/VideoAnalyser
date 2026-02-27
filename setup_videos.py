"""
Video Download Helper Script
Downloads sample art/marketing videos for the GenTA pipeline.
Supports public-domain sources (Pexels, Pixabay).
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_downloader")

# Public domain video sources
VIDEO_SOURCES = {
    "pexels": {
        "api": "https://api.pexels.com/videos/search",
        "requires_key": True,
        "docs": "https://www.pexels.com/api/documentation/"
    },
    "pixabay": {
        "api": "https://pixabay.com/api/videos/",
        "requires_key": True,
        "docs": "https://pixabay.com/api/docs/"
    },
    "archive_org": {
        "collections": [
            "community_video",
            "open_source_movies",
            "feature_films"
        ],
        "api": "https://archive.org/advancedsearch.php",
        "requires_key": False
    }
}

SAMPLE_VIDEOS = {
    "abstract_art": {
        "description": "Abstract artistic visual with motion graphics",
        "search_terms": ["abstract", "art", "motion"],
        "duration": "30s-2m"
    },
    "marketing_creative": {
        "description": "Marketing/advertising style dynamic creative",
        "search_terms": ["marketing", "advertising", "product"],
        "duration": "15s-1m"
    },
    "contemporary_art": {
        "description": "Contemporary art exhibition footage",
        "search_terms": ["art", "exhibition", "gallery"],
        "duration": "30s-3m"
    }
}


def setup_video_directory() -> Path:
    """Create videos directory if it doesn't exist."""
    video_dir = Path("videos")
    video_dir.mkdir(exist_ok=True)
    logger.info(f"✓ Video directory ready: {video_dir.resolve()}")
    return video_dir


def print_download_instructions():
    """Print instructions for manually downloading videos."""
    print("\n" + "="*80)
    print("VIDEO DOWNLOAD INSTRUCTIONS")
    print("="*80 + "\n")
    print("""
This script helps you download sample videos for the GenTA pipeline.
The pipeline works with ANY short videos, but here are recommended sources:

📺 RECOMMENDED VIDEO SOURCES (Public Domain / CC Licensed):

1. PEXELS VIDEOS (Free, no attribution required)
   - Site: https://www.pexels.com/videos/
   - Search: "abstract", "marketing", "art", "motion"
   - Download: Save .mp4 files → ./videos/ folder
   - Example: https://www.pexels.com/video/

2. PIXABAY VIDEOS (Free, Pixabay License)
   - Site: https://pixabay.com/videos/
   - Search: "abstract art", "video marketing", "contemporary"
   - Download: Save .mp4 files → ./videos/ folder
   - Example: https://pixabay.com/videos/

3. ARCHIVE.ORG (Public Domain Films)
   - Site: https://archive.org/details/movies
   - Collections: "open_source_movies", "community_video"
   - Download: Use Internet Archive downloader or direct links
   - Example: https://archive.org/details/opensourcemovies

🎬 RECOMMENDED VIDEO SPECIFICATIONS:

For optimal pipeline performance:
✓ Format: MP4 or AVI
✓ Length: 30 seconds to 2 minutes
✓ Resolution: 720p or higher recommended
✓ Content: Art, marketing creatives, music videos, animations
✓ Count: 2-3 videos to start

📥 HOW TO USE:

1. Visit one of the sources above
2. Search for videos (e.g., "abstract", "marketing", "art")
3. Download 2-3 videos as .mp4 files
4. Save them in the ./videos/ folder
5. Run the pipeline:
   
   python extract_frames.py
   python embed_frames.py
   python similarity_heatmap.py

💡 EXAMPLE VIDEO SEARCHES:

For Abstract/Artistic Content:
- "abstract animation"
- "motion graphics"
- "digital art"
- "light effects"

For Marketing Content:
- "product marketing"
- "advertising video"
- "commercial"
- "promotion"

🔗 QUICK LINKS:

Pexels: https://www.pexels.com/videos/
Pixabay: https://pixabay.com/videos/
Archive.org Movies: https://archive.org/details/movies

""")
    print("="*80 + "\n")


def check_existing_videos():
    """Check for existing videos in ./videos/ directory."""
    video_dir = Path("videos")
    
    if not video_dir.exists():
        return []
    
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if video_files:
        print("\n✓ Found existing videos:")
        for vf in video_files:
            size_mb = vf.stat().st_size / (1024 * 1024)
            print(f"  • {vf.name} ({size_mb:.1f} MB)")
        return video_files
    else:
        print("\n⚠️  No videos found in ./videos/ directory")
        return []


def create_sample_config():
    """Create a sample configuration file."""
    config = {
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
        },
        "video_sources": {
            "recommended": list(VIDEO_SOURCES.keys()),
            "instructions": "See README.md and setup_videos.py for details"
        }
    }
    
    config_path = Path("config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Sample config created: {config_path}")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("GenTA Pipeline: Video Setup Helper")
    print("="*80 + "\n")
    
    # Create video directory
    video_dir = setup_video_directory()
    
    # Check for existing videos
    existing = check_existing_videos()
    
    if not existing:
        # Print instructions
        print_download_instructions()
        
        print("\n📋 NEXT STEPS:\n")
        print("1. Visit one of the video sources listed above")
        print("2. Download 2-3 short videos (MP4 format)")
        print("3. Save them to: ./videos/ folder")
        print("4. Then run: python extract_frames.py\n")
    else:
        print("\n✓ Videos ready! Run the pipeline:")
        print("  python extract_frames.py")
        print("  python embed_frames.py")
        print("  python similarity_heatmap.py\n")
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "="*80)
    print("📖 For more details, see:")
    print("  • README.md - Full documentation")
    print("  • GenTA_Affective_Computing_Pipeline.ipynb - Interactive walkthrough")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
