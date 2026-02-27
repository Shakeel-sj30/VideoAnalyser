"""
Vibe Similarity Analysis & Visualization
Computes frame-to-frame similarity using CLIP embeddings.
Generates heatmaps and retrieves most similar frames (mood/style matches).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VibeSimilarityAnalyzer:
    """Analyze mood/style similarity between video frames."""
    
    def __init__(
        self,
        embeddings_path: str = "embeddings/frame_embeddings.npy",
        metadata_path: str = "embeddings/embeddings_metadata.json"
    ):
        """Load embeddings and metadata."""
        embeddings_path = Path(embeddings_path)
        metadata_path = Path(metadata_path)
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
        
        self.embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(self.metadata['index_mapping'])} frames")
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.embeddings)
        logger.info(f"Similarity matrix computed: {self.similarity_matrix.shape}")
    
    def get_filename(self, index: int) -> str:
        """Get filename for frame at given index."""
        if self.metadata and index < len(self.metadata["index_mapping"]):
            return self.metadata["index_mapping"][index]["filename"]
        return f"frame_{index}"
    
    def get_top_k_similar(
        self, 
        query_index: int, 
        k: int = 5,
        include_query: bool = False
    ) -> List[Tuple[int, float, str]]:
        """
        Get top-k most similar frames to a query frame.
        
        Args:
            query_index: Index of query frame
            k: Number of similar frames to return
            include_query: Whether to include query itself
            
        Returns:
            List of (index, similarity_score, filename) tuples
        """
        if query_index >= len(self.embeddings):
            raise ValueError(f"Query index {query_index} out of range")
        
        similarities = self.similarity_matrix[query_index]
        
        # Get top indices (excluding query if requested)
        if include_query:
            top_indices = np.argsort(similarities)[::-1][:k]
        else:
            top_indices = np.argsort(similarities)[::-1][:k+1]
            # Filter out query index
            top_indices = top_indices[top_indices != query_index][:k]
        
        results = [
            (idx, similarities[idx], self.get_filename(idx))
            for idx in top_indices
        ]
        
        return results
    
    def analyze_query_frames(
        self, 
        query_indices: List[int],
        k: int = 5
    ) -> Dict:
        """
        Analyze top-k similar frames for multiple query frames.
        This simulates the "vibe matching" for mood/style similarity.
        
        Args:
            query_indices: List of frame indices to query
            k: Number of similar frames per query
            
        Returns:
            Analysis dictionary with results
        """
        results = {}
        
        for query_idx in query_indices:
            if query_idx >= len(self.embeddings):
                logger.warning(f"Query index {query_idx} out of range, skipping")
                continue
            
            similar_frames = self.get_top_k_similar(query_idx, k=k)
            
            results[f"query_{query_idx}"] = {
                "query_frame": self.get_filename(query_idx),
                "query_index": int(query_idx),
                "similar_frames": [
                    {
                        "index": int(idx),
                        "filename": fname,
                        "similarity_score": float(sim)
                    }
                    for idx, sim, fname in similar_frames
                ]
            }
            
            logger.info(f"Query {query_idx}: {self.get_filename(query_idx)}")
            for idx, sim, fname in similar_frames:
                logger.info(f"  ↳ {fname}: {sim:.4f}")
        
        return results
    
    def visualize_heatmap(
        self,
        output_path: str = "embeddings/similarity_heatmap.png",
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "YlOrRd"
    ):
        """
        Create and save similarity heatmap visualization.
        
        Args:
            output_path: Where to save the figure
            figsize: Figure size (width, height)
            cmap: Matplotlib colormap name
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(self.similarity_matrix, cmap=cmap, aspect='auto')
        
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Frame Index", fontsize=12)
        ax.set_title("Frame-to-Frame Vibe Similarity Heatmap\n(CLIP Embeddings - Cosine Similarity)", 
                     fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Similarity Score", fontsize=11)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Heatmap saved to {output_path}")
        
        plt.close()
    
    def visualize_query_results(
        self,
        query_indices: List[int],
        k: int = 5,
        output_path: str = "embeddings/query_results.png"
    ):
        """
        Visualize query results as subplots showing top-k similar frames.
        
        Args:
            query_indices: Query frame indices
            k: Number of similar frames to show
            output_path: Where to save figure
        """
        # Note: This requires actual frame images to be available
        # For now, we'll create a text-based visualization
        logger.info("Creating query results report...")
        
        # This is a placeholder for when frame images are available
        logger.info(f"Would visualize {len(query_indices)} queries with top {k} results")
    
    def generate_report(
        self,
        query_indices: List[int] = None,
        k: int = 5,
        output_path: str = "embeddings/similarity_report.json"
    ) -> Dict:
        """
        Generate comprehensive similarity analysis report.
        
        Args:
            query_indices: Frames to analyze (if None, uses first 3)
            k: Number of similar frames per query
            output_path: Where to save report
            
        Returns:
            Report dictionary
        """
        if query_indices is None:
            query_indices = list(range(min(3, len(self.embeddings))))
        
        # Analyze queries
        query_results = self.analyze_query_frames(query_indices, k=k)
        
        # Compute global stats
        similarities_flat = self.similarity_matrix[
            np.triu_indices_from(self.similarity_matrix, k=1)
        ]
        
        report = {
            "analysis_type": "Vibe Similarity Analysis",
            "total_frames": int(len(self.embeddings)),
            "embedding_dimension": int(self.embeddings.shape[1]),
            "global_statistics": {
                "similarity_mean": float(similarities_flat.mean()),
                "similarity_std": float(similarities_flat.std()),
                "similarity_min": float(similarities_flat.min()),
                "similarity_max": float(similarities_flat.max()),
                "similarity_median": float(np.median(similarities_flat))
            },
            "query_results": query_results,
            "interpretation": {
                "high_similarity": "Frames with similar mood/style aesthetic",
                "low_similarity": "Frames with contrasting mood/style",
                "use_case": "For GenTA's affective computing: identify emotionally/aesthetically similar marketing creatives"
            }
        }
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        return report
    
    def print_statistics(self):
        """Print summary statistics."""
        similarities_flat = self.similarity_matrix[
            np.triu_indices_from(self.similarity_matrix, k=1)
        ]
        
        print("\n" + "="*60)
        print("VIBE SIMILARITY STATISTICS")
        print("="*60)
        print(f"Total frames: {len(self.embeddings)}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        print(f"\nSimilarity Score Distribution (0-1 scale):")
        print(f"  Mean:     {similarities_flat.mean():.4f}")
        print(f"  Std Dev:  {similarities_flat.std():.4f}")
        print(f"  Min:      {similarities_flat.min():.4f}")
        print(f"  Max:      {similarities_flat.max():.4f}")
        print(f"  Median:   {np.median(similarities_flat):.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = VibeSimilarityAnalyzer()
        
        # Print statistics
        analyzer.print_statistics()
        
        # Generate visualizations and reports
        analyzer.visualize_heatmap()
        
        query_frames = [0, len(analyzer.embeddings)//2, len(analyzer.embeddings)-1]
        report = analyzer.generate_report(query_indices=query_frames, k=5)
        
        print("\n✓ Similarity analysis complete!")
        print(f"Outputs saved to: embeddings/")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise