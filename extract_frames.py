"""
Frame Extraction Pipeline
Extracts representative frames from video files using interval-based sampling.
Saves frames with metadata for downstream processing.
"""

import cv2
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FrameExtractor:
    """Extract frames from video files and save with metadata."""
    
    def __init__(self, video_dir: str = "videos", output_dir: str = "frames"):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metadata = {}
    
    def extract_frames(
        self, 
        video_path: str, 
        frame_interval_seconds: float = 1.0,
        video_id: str = None
    ) -> Tuple[int, List[str]]:
        """
        Extract frames from a single video at specified intervals.
        
        Args:
            video_path: Path to video file
            frame_interval_seconds: Seconds between extracted frames
            video_id: Identifier for this video (used for metadata)
            
        Returns:
            Tuple of (number_of_frames_saved, list_of_filenames)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if video_id is None:
            video_id = video_path.stem
        
        logger.info(f"Extracting frames from: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Verify video was opened successfully
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise IOError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video stats - FPS: {fps}, Total frames: {total_frames}, Resolution: {width}x{height}")
        
        frame_interval = int(fps * frame_interval_seconds)
        frame_count = 0
        saved = 0
        frame_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % max(1, frame_interval) == 0:
                filename = f"{video_id}_frame_{saved:04d}.jpg"
                filepath = self.output_dir / filename
                
                # Verify frame is valid before saving
                if frame is not None and frame.size > 0:
                    cv2.imwrite(str(filepath), frame)
                    timestamp = frame_count / fps
                    
                    frame_list.append({
                        "filename": filename,
                        "filepath": str(filepath),
                        "video_id": video_id,
                        "frame_index": frame_count,
                        "timestamp_seconds": timestamp,
                        "local_index": saved
                    })
                    
                    saved += 1
            
            frame_count += 1
        
        cap.release()
        
        # Store metadata for this video
        self.metadata[video_id] = {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "resolution": [width, height],
            "extracted_frames": saved,
            "frame_interval_seconds": frame_interval_seconds,
            "frames": frame_list
        }
        
        logger.info(f"Saved {saved} frames from {video_id}")
        return saved, [f["filename"] for f in frame_list]
    
    def extract_all_videos(
        self, 
        frame_interval_seconds: float = 1.0
    ) -> Dict[str, Tuple[int, List[str]]]:
        """Extract frames from all MP4 videos in videos directory."""
        results = {}
        
        video_files = list(self.video_dir.glob("*.mp4"))
        if not video_files:
            logger.warning(f"No MP4 files found in {self.video_dir}")
            return results
        
        for video_file in sorted(video_files):
            try:
                saved, filenames = self.extract_frames(
                    str(video_file),
                    frame_interval_seconds
                )
                results[video_file.stem] = (saved, filenames)
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
        
        return results
    
    def save_metadata(self, output_path: str = "frames/metadata.json"):
        """Save frame metadata to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    # Extract frames from all videos
    extractor = FrameExtractor()
    
    # Process all MP4 files in videos directory
    results = extractor.extract_all_videos(frame_interval_seconds=1.0)
    
    # Save metadata
    extractor.save_metadata()
    
    logger.info(f"Frame extraction complete. Summary: {results}")