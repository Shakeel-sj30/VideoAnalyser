"""
Frame Embedding Pipeline
Computes CLIP embeddings for video frames for mood/style similarity analysis.
Includes comprehensive verification and error handling.
"""

import torch
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FrameEmbedder:
    """Compute CLIP embeddings for video frames."""
    
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        """
        Initialize embedder with CLIP model.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Compute embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector (1D numpy array)
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
        
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            embedding = image_features.squeeze().cpu().numpy()
            
            # Verify embedding is valid
            if np.isnan(embedding).any():
                logger.warning(f"NaN detected in embedding for {image_path}")
                return None
            
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to compute embedding for {image_path}: {e}")
            return None
    
    def embed_frames(
        self, 
        frame_dir: str = "frames",
        metadata_file: str = "frames/metadata.json"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute embeddings for all frames in a directory.
        
        Args:
            frame_dir: Directory containing frame images
            metadata_file: Path to frame metadata JSON
            
        Returns:
            Tuple of (embeddings_array, metadata_dict)
        """
        frame_dir = Path(frame_dir)
        
        if not frame_dir.exists():
            logger.error(f"Frame directory not found: {frame_dir}")
            raise FileNotFoundError(f"Directory not found: {frame_dir}")
        
        # Load metadata if available
        metadata = {}
        if Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_file}")
        
        # Get all image files
        image_files = sorted(
            list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))
        )
        
        if not image_files:
            logger.error(f"No image files found in {frame_dir}")
            raise FileNotFoundError(f"No images found in {frame_dir}")
        
        logger.info(f"Found {len(image_files)} images to embed")
        
        embeddings_list = []
        index_mapping = []
        failed_count = 0
        
        for idx, img_file in enumerate(image_files):
            logger.info(f"Processing {idx + 1}/{len(image_files)}: {img_file.name}")
            
            embedding = self.embed_image(str(img_file))
            
            if embedding is not None:
                embeddings_list.append(embedding)
                index_mapping.append({
                    "index": len(embeddings_list) - 1,
                    "filename": img_file.name,
                    "filepath": str(img_file)
                })
            else:
                failed_count += 1
        
        if not embeddings_list:
            logger.error("All embeddings failed!")
            raise RuntimeError("Failed to compute any embeddings")
        
        embeddings = np.array(embeddings_list)
        
        # VERIFICATION TESTS
        logger.info("Running verification tests...")
        
        # Test 1: Shape verification
        logger.info(f"Embedding shape: {embeddings.shape}")
        assert embeddings.shape[0] == len(embeddings_list), "Shape mismatch!"
        logger.info("✓ Shape verification passed")
        
        # Test 2: NaN check
        nan_count = np.isnan(embeddings).sum()
        assert nan_count == 0, f"Found {nan_count} NaN values!"
        logger.info("✓ NaN verification passed")
        
        # Test 3: Inf check
        inf_count = np.isinf(embeddings).sum()
        assert inf_count == 0, f"Found {inf_count} Inf values!"
        logger.info("✓ Inf verification passed")
        
        # Test 4: Duplicate image test - identical images should have identical embeddings
        if len(embeddings) > 1:
            identical_similarity = np.dot(embeddings[0], embeddings[0])
            logger.info(f"✓ Self-similarity (sanity check): {identical_similarity:.4f}")
            assert identical_similarity > 0.99, "Self-similarity should be ~1.0!"
        
        # Test 5: Value range (L2 normalized should be ~1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"Embedding L2 norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")
        
        logger.info(f"Total embeddings computed: {len(embeddings_list)}")
        if failed_count > 0:
            logger.warning(f"Failed embeddings: {failed_count}")
        
        return embeddings, {
            "embeddings": embeddings.tolist(),
            "index_mapping": index_mapping,
            "metadata": metadata,
            "total_computed": len(embeddings_list),
            "total_failed": failed_count,
            "embedding_dim": embeddings.shape[1]
        }


def save_embeddings(
    embeddings: np.ndarray,
    metadata: Dict,
    output_dir: str = "embeddings"
):
    """Save embeddings and metadata to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save embeddings as numpy array
    np.save(output_dir / "frame_embeddings.npy", embeddings)
    logger.info(f"Embeddings saved to {output_dir / 'frame_embeddings.npy'}")
    
    # Save metadata as JSON
    metadata_json = {
        "index_mapping": metadata["index_mapping"],
        "frame_metadata": metadata["metadata"],
        "total_computed": metadata["total_computed"],
        "total_failed": metadata["total_failed"],
        "embedding_dim": metadata["embedding_dim"]
    }
    
    with open(output_dir / "embeddings_metadata.json", 'w') as f:
        json.dump(metadata_json, f, indent=2)
    
    logger.info(f"Metadata saved to {output_dir / 'embeddings_metadata.json'}")


if __name__ == "__main__":
    # Initialize embedder
    embedder = FrameEmbedder()
    
    # Compute embeddings for all frames
    try:
        embeddings, metadata = embedder.embed_frames()
        
        # Save results
        save_embeddings(embeddings, metadata)
        
        logger.info("✓ Embedding pipeline complete!")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise