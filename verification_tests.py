"""
Verification & Testing Suite for Affective Computing Pipeline
Provides unit tests and assertions for pipeline robustness.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class PipelineVerifier:
    """Verification suite for pipeline components."""
    
    @staticmethod
    def verify_embeddings(
        embeddings: np.ndarray,
        min_samples: int = 5,
        strict: bool = True
    ) -> Tuple[bool, str]:
        """
        Verify embeddings array integrity.
        
        Args:
            embeddings: Embeddings array
            min_samples: Minimum number of samples expected
            strict: If True, fail on any warning
            
        Returns:
            Tuple of (is_valid, message)
        """
        issues = []
        
        # Test 1: Shape
        if embeddings.ndim != 2:
            issues.append(f"Embeddings must be 2D, got {embeddings.ndim}D")
        
        # Test 2: Minimum samples
        if embeddings.shape[0] < min_samples:
            issues.append(f"Expected >= {min_samples} samples, got {embeddings.shape[0]}")
        
        # Test 3: NaN values
        nan_count = np.isnan(embeddings).sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
        
        # Test 4: Inf values
        inf_count = np.isinf(embeddings).sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} Inf values")
        
        # Test 5: All zeros
        zero_rows = np.all(embeddings == 0, axis=1).sum()
        if zero_rows > 0:
            issues.append(f"Found {zero_rows} all-zero embeddings")
        
        # Test 6: Self-similarity (diagonal should be ~1 for normalized embeddings)
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms < 0.9) or np.any(norms > 1.1):
            issues.append(f"Embeddings may not be properly normalized (norms: {norms.min():.4f}-{norms.max():.4f})")
        
        is_valid = len(issues) == 0 or (not strict and all("may not" in i for i in issues))
        message = "; ".join(issues) if issues else "✓ All checks passed"
        
        return is_valid, message
    
    @staticmethod
    def verify_similarity_matrix(
        similarity: np.ndarray,
        embeddings: np.ndarray = None
    ) -> Tuple[bool, str]:
        """
        Verify similarity/distance matrix validity.
        
        Args:
            similarity: Similarity matrix
            embeddings: Optional embeddings for cross-check
            
        Returns:
            Tuple of (is_valid, message)
        """
        issues = []
        
        # Test 1: Square
        if similarity.shape[0] != similarity.shape[1]:
            issues.append("Similarity matrix must be square")
        
        # Test 2: Value range (cosine similarity should be -1 to 1)
        if similarity.min() < -1.001 or similarity.max() > 1.001:
            issues.append(f"Cosine similarity out of range: [{similarity.min():.4f}, {similarity.max():.4f}]")
        
        # Test 3: Symmetry (should be symmetric for cosine distance)
        if not np.allclose(similarity, similarity.T, atol=1e-5):
            issues.append("Similarity matrix is not symmetric")
        
        # Test 4: Diagonal check
        diag = np.diag(similarity)
        if not np.allclose(diag, 1.0, atol=1e-5):
            issues.append(f"Diagonal values not ~1.0: {diag}")
        
        # Test 5: No NaNs
        if np.isnan(similarity).any():
            issues.append("Similarity matrix contains NaN values")
        
        is_valid = len(issues) == 0
        message = "; ".join(issues) if issues else "✓ All checks passed"
        
        return is_valid, message
    
    @staticmethod
    def test_embedding_uniqueness(
        embeddings: np.ndarray,
        similarity_threshold: float = 0.99
    ) -> Tuple[int, str]:
        """
        Check for potentially duplicate embeddings.
        
        Args:
            embeddings: Embeddings array
            similarity_threshold: Threshold for flagging as duplicate
            
        Returns:
            Tuple of (duplicate_count, message)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity = cosine_similarity(embeddings)
        
        # Check for duplicates (excluding self-similarity)
        duplicate_pairs = 0
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                if similarity[i, j] > similarity_threshold:
                    duplicate_pairs += 1
        
        message = f"Found {duplicate_pairs} potential duplicate pairs"
        return duplicate_pairs, message
    
    @staticmethod
    def test_frame_consistency(
        metadata_file: str,
        frame_dir: str
    ) -> Tuple[bool, str]:
        """
        Verify frame metadata matches actual files.
        
        Args:
            metadata_file: Path to metadata JSON
            frame_dir: Frame directory path
            
        Returns:
            Tuple of (is_consistent, message)
        """
        issues = []
        
        metadata_path = Path(metadata_file)
        frame_path = Path(frame_dir)
        
        if not metadata_path.exists():
            return False, "Metadata file not found"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check index mapping
        if "index_mapping" in metadata:
            for entry in metadata["index_mapping"]:
                filepath = Path(entry["filepath"])
                if not filepath.exists():
                    issues.append(f"Referenced file missing: {filepath}")
        
        # Check actual files
        actual_files = set(f.name for f in frame_path.glob("*.jpg") + frame_path.glob("*.png"))
        
        if "index_mapping" in metadata:
            mapped_files = set(e["filename"] for e in metadata["index_mapping"])
            missing = actual_files - mapped_files
            extra = mapped_files - actual_files
            
            if missing:
                issues.append(f"Files not in metadata: {missing}")
            if extra:
                issues.append(f"Metadata files not found: {extra}")
        
        is_valid = len(issues) == 0
        message = "; ".join(issues) if issues else "✓ Frames and metadata consistent"
        
        return is_valid, message


def run_full_validation(
    embeddings_path: str = "embeddings/frame_embeddings.npy",
    metadata_path: str = "embeddings/embeddings_metadata.json",
    frame_dir: str = "frames"
) -> Dict[str, Tuple[bool, str]]:
    """
    Run full validation suite on pipeline outputs.
    
    Returns:
        Dictionary of validation results
    """
    print("\n" + "="*70)
    print("RUNNING FULL VALIDATION SUITE")
    print("="*70 + "\n")
    
    results = {}
    verifier = PipelineVerifier()
    
    # Test 1: Embeddings existence
    if Path(embeddings_path).exists():
        embeddings = np.load(embeddings_path)
        
        is_valid, msg = verifier.verify_embeddings(embeddings)
        results["embeddings_integrity"] = (is_valid, msg)
        print(f"Embeddings Integrity: {'✓ PASS' if is_valid else '✗ FAIL'}")
        print(f"  {msg}\n")
        
        is_valid, msg = verifier.verify_similarity_matrix(
            np.corrcoef(embeddings.T)
        )
        results["similarity_matrix"] = (is_valid, msg)
        print(f"Similarity Matrix: {'✓ PASS' if is_valid else '✗ FAIL'}")
        print(f"  {msg}\n")
        
        dup_count, msg = verifier.test_embedding_uniqueness(embeddings)
        results["embedding_uniqueness"] = (dup_count < 5, msg)
        print(f"Embedding Uniqueness: {'✓ PASS' if dup_count < 5 else '⚠ WARNING'}")
        print(f"  {msg}\n")
    else:
        print(f"⚠ Embeddings not found at {embeddings_path}\n")
    
    # Test 2: Metadata consistency
    is_valid, msg = verifier.test_frame_consistency(metadata_path, frame_dir)
    results["frame_consistency"] = (is_valid, msg)
    print(f"Frame Consistency: {'✓ PASS' if is_valid else '✗ FAIL'}")
    print(f"  {msg}\n")
    
    print("="*70)
    all_pass = all(result[0] for result in results.values())
    print(f"Overall Status: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    run_full_validation()
