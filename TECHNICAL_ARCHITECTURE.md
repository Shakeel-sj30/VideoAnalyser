# Technical Architecture Document
## GenTA Mini GACS Prototype: Mood & Style Embedding Pipeline

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│  Videos (MP4, AVI) - Art/Marketing Content                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: FRAME EXTRACTION                       │
│  • OpenCV VideoCapture                                      │
│  • Interval-based sampling (1 frame per N seconds)         │
│  • Metadata tracking (timestamp, video_id, local_index)    │
│  • Output: JPEG images + metadata.json                     │
└────────────┬────────────────────────────────────────────────┘
             │
    frames/ ▼ metadata.json
             │
┌─────────────────────────────────────────────────────────────┐
│          STAGE 2: EMBEDDING GENERATION                       │
│  • HuggingFace: openai/clip-vit-base-patch32              │
│  • Input: RGB images                                        │
│  • Output: 512-dimensional vectors (mood/style vectors)    │
│  • GPU acceleration: 50ms/frame (CPU: 150ms/frame)        │
└────────────┬────────────────────────────────────────────────┘
             │
embeddings/  ▼ frame_embeddings.npy (N×512 array)
             │
┌─────────────────────────────────────────────────────────────┐
│         STAGE 3: SIMILARITY ANALYSIS                        │
│  • Cosine similarity: pairwise distances (N×N matrix)      │
│  • Top-K retrieval: find similar frames per query          │
│  • Statistics: mean, std, percentiles of similarities      │
└────────────┬────────────────────────────────────────────────┘
             │
outputs/     ▼ similarity_report.json
             │
┌─────────────────────────────────────────────────────────────┐
│       STAGE 4: VISUALIZATION & REPORTING                   │
│  • Heatmap: frame-to-frame similarity matrix              │
│  • 2D Projection: PCA embedding visualization             │
│  • Bar Charts: top-k results per query frame              │
│  • JSON export: structured results                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│  • PNG heatmaps (similarity_heatmap.png)                  │
│  • 2D visualization (embedding_projection_2d.png)         │
│  • Query results (query_results_bars.png)                 │
│  • Report JSON (similarity_report.json)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 FrameExtractor (extract_frames.py)

**Class:** `FrameExtractor`

**Purpose:** Extract and catalog frames from video files

**Key Methods:**
```python
extract_frames(video_path, frame_interval_seconds, video_id)
    → (saved_count, frame_metadata_list)
    
extract_all_videos(frame_interval_seconds)
    → {video_id: (saved_count, filenames), ...}
    
save_metadata(output_path)
    → Saves JSON mapping
```

**Design Decisions:**
- ✓ Interval-based (not scene detection) → deterministic & reproducible
- ✓ Metadata tracking → enables traceability throughout pipeline
- ✓ Per-video organization → supports multi-video datasets
- ✓ Graceful error handling → skips missing/bad videos

**Data Flow:**
```
VideoFile
  ▼ (OpenCV)
FrameCapture ─→ FrameValidation ─→ ImageWrite + MetadataRecord
  ▼
JSON Export
```

---

### 2.2 FrameEmbedder (embed_frames.py)

**Class:** `FrameEmbedder`

**Purpose:** Generate mood/style embeddings using CLIP

**Key Methods:**
```python
embed_image(image_path)
    → np.ndarray (512,)  # Single embedding
    
embed_frames(frame_dir, metadata_file)
    → (embeddings_array, metadata_dict)
```

**Model Selection: CLIP-ViT-B32**

| Property | Value | Rationale |
|----------|-------|-----------|
| Architecture | Vision Transformer | Robust to image variations |
| Pretrain Data | 400M image-text pairs | Learns semantic relationships |
| Output Dim | 512 | Balance: expressiveness vs. efficiency |
| Speed | 50-150ms/image | Reasonable real-time performance |
| License | OpenAI (research use) | Academic/commercial flexibility |

**Verification Test Suite:**

```python
Test 1: Shape Consistency
    Assert: embeddings.shape == (N_samples, 512)
    
Test 2: Value Integrity
    Assert: not np.isnan(embeddings).any()
    Assert: not np.isinf(embeddings).any()
    
Test 3: Normalization
    norms = np.linalg.norm(embeddings, axis=1)
    Assert: 0.95 < norms.mean() < 1.05  # L2 normalized
    
Test 4: Self-Similarity (Identity Property)
    sim_matrix = cosine_similarity(embeddings)
    Assert: np.allclose(np.diag(sim_matrix), 1.0, atol=1e-5)
    
Test 5: Embedding Diversity
    upper_tri = pairwise_sim[np.triu_indices(..., k=1)]
    Assert: upper_tri.std() > 0.1  # Non-degenerate
    
Test 6: Duplicate Detection
    dup_count = np.sum(upper_tri > 0.99)
    Log: "Found {dup_count} potential duplicate pairs"
```

**GPU/CPU Support:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Auto-detects & adapts
# GPU: ~50ms/image
# CPU: ~150ms/image
```

---

### 2.3 VibeSimilarityAnalyzer (similarity_heatmap.py)

**Class:** `VibeSimilarityAnalyzer`

**Purpose:** Compute & interpret frame-to-frame mood similarity

**Key Methods:**
```python
get_top_k_similar(query_index, k=5)
    → [(index, similarity, filename), ...]
    
analyze_query_frames(query_indices, k=5)
    → {query_results with top-k per frame}
    
visualize_heatmap(output_path, cmap="YlOrRd")
    → Saves PNG of similarity matrix
    
generate_report(query_indices=None, k=5)
    → Structured JSON report
```

**Similarity Metric: Cosine Distance**

Why cosine similarity?
```
S(u, v) = (u · v) / (||u|| ||v||)

Properties:
✓ Range: [-1, 1]
✓ Normalized embeddings → S(u,u) = 1.0 exactly
✓ Symmetric: S(u,v) = S(v,u)
✓ Interpretation: Angle between vectors
✓ Invariant to: Magnitude (semantic similarity)
```

**Query-Based Retrieval:**
```python
Similarity Matrix: N×N (cosine distances)
Query Index: i (frame to find similar matches for)
Top-K: Query row i, exclude self-match, take max k

Result: [(idx1, sim1, name1), (idx2, sim2, name2), ...]
        where sim1 >= sim2 >= ... >= simk
```

---

### 2.4 Verification Suite (verification_tests.py)

**Class:** `PipelineVerifier`

**Purpose:** Validate pipeline outputs at each stage

**Test Categories:**

1. **Embedding Integrity Tests**
   - Shape validation (N×512)
   - NaN/Inf detection
   - Normalization check (L2 norm ≈ 1.0)
   - All-zero detection

2. **Similarity Matrix Tests**
   - Symmetry: S[i,j] = S[j,i]
   - Diagonal: S[i,i] = 1.0
   - Range: [-1, 1]
   - No NaN propagation

3. **Frame Consistency Tests**
   - Metadata file → actual files
   - No orphaned files
   - Index mapping integrity

4. **Uniqueness Tests**
   - Duplicate pair detection
   - Diversity scoring

---

## 3. Data Formats

### 3.1 Frame Metadata (JSON)

```json
{
  "extraction_timestamp": "2025-02-26T10:30:00",
  "total_frames": 120,
  "frames": [
    {
      "filename": "video1_frame_0000.jpg",
      "filepath": "/absolute/path/to/frame.jpg",
      "video_id": "video1",
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "local_index": 0
    },
    ...
  ]
}
```

### 3.2 Embeddings Array (NumPy)

```python
shape: (N_frames, 512)
dtype: float32
values: [-1.0, 1.0]  # For normalized embeddings
```

### 3.3 Similarity Report (JSON)

```json
{
  "query_0": {
    "query_frame": "video1_frame_0000.jpg",
    "query_index": 0,
    "similar_frames": [
      {
        "index": 5,
        "filename": "video1_frame_0005.jpg",
        "similarity_score": 0.876
      },
      ...
    ]
  },
  ...
}
```

---

## 4. Computational Complexity

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Frame extraction | O(V×F) | V=videos, F=frames per video |
| Embedding computation | O(N×M) | N=frames, M=embedding time |
| Similarity matrix | O(N²) | Pairwise distance (unavoidable) |
| Top-K retrieval | O(N log K) | Per query (sorting) |
| Visualization | O(N²) | Heatmap rendering |

### Memory Complexity

| Component | Size | Notes |
|-----------|------|-------|
| Embeddings | N×512×4B | N=frames, 512-dim float32 |
| Similarity matrix | N²×4B | Dense matrix |
| Frames (on disk) | N×50KB | JPEG compressed |
| Metadata | ~1KB per frame | JSON ext

---

## 5. Error Handling Strategy

### Error Cascade Prevention

```
Recoverable Errors (Log + Continue)
  - Missing frame file → skip frame
  - Invalid embedding → skip frame
  - Corrupted image → skip image

Critical Errors (Fail Fast)
  - No videos found → exit with guidance
  - No embeddings generated → exit
  - Empty frame directory → exit

Warnings (Log + Proceed)
  - Potential duplicates → log count
  - Non-normalized embeddings → warn
```

---

## 6. Scalability Considerations

### Current Limits
- **Frames:** 0-10,000 (GPU memory)
- **Videos:** 1-100 (depends on frame rate)
- **Embedding Dim:** 512 (CLIP-ViT-B32 fixed)

### Scalability Paths

**Path 1: Distributed Processing**
```
Video1 → Extract → Embed → Queue
Video2 → Extract → Embed → Queue
Video3 → Extract → Embed → Queue
         ↓         ↓         ↓
    Async Processing Pipeline
         ↓
   Aggregated Results
```

**Path 2: Model Optimization**
```
CLIP-ViT-B32 (current)  → 512-dim, 50ms/frame
↓
Quantization (INT8)     → 256-dim, 20ms/frame
↓
Distillation (smaller)  → 128-dim, 10ms/frame
```

---

## 7. Integration Points (Future GACS)

### Performance Feedback Loop

```
Frames + Embeddings
    ↓
Similarity Scores
    ↓
KPI Labels (CTR, CVR, ROAS)  ← From A/B tests
    ↓
Training Data
    ↓
Regression Model: Similarity → KPI Prediction
    ↓
Real-time Scoring API
```

### Human-in-the-Loop

```
Designer: Generate Creative Variations
    ↓
Pipeline: Score by predicted KPI
    ↓
Ranking: Top-k suggestions
    ↓
Human: Approve/Reject
    ↓
Feedback: Retrain predictor
    ↓
Organizational Intelligence: Build aesthetic knowledge
```

---

## 8. Testing Strategy

### Unit Tests

```python
test_frame_extraction()
    # Load video → extract frames → verify count
    
test_embedding_computation()
    # Load image → compute embedding → verify shape
    
test_similarity_computation()
    # Create embeddings → compute similarity → verify symmetry
```

### Integration Tests

```python
test_end_to_end_pipeline()
    # Sample video → all 4 stages → verify outputs exist
    
test_error_handling()
    # Missing files, corrupted images, invalid embeddings
```

### Verification Tests

```python
test_embedding_integrity()
test_similarity_matrix_properties()
test_frame_consistency()
test_duplicate_detection()
```

---

## 9. Deployment Checklist

- [ ] All tests pass (unit + integration + verification)
- [ ] Dependencies locked (requirements.txt)
- [ ] Error handling in place
- [ ] Logging configured
- [ ] Documentation complete
- [ ] README has setup instructions
- [ ] Edge cases handled (empty dirs, missing videos)
- [ ] Performance profiled (timing logged)
- [ ] Reproducibility verified (same output twice)

---

## 10. API Examples

### Simple Pipeline Execution

```python
from extract_frames import FrameExtractor
from embed_frames import FrameEmbedder, save_embeddings
from similarity_heatmap import VibeSimilarityAnalyzer

# Stage 1: Extract
extractor = FrameExtractor()
extractor.extract_all_videos(frame_interval_seconds=1.0)
extractor.save_metadata()

# Stage 2: Embed
embedder = FrameEmbedder()
embeddings, metadata = embedder.embed_frames()
save_embeddings(embeddings, metadata)

# Stage 3: Analyze
analyzer = VibeSimilarityAnalyzer()
report = analyzer.generate_report(query_indices=[0, 50, 100])
analyzer.visualize_heatmap()
```

---

## 11. Future Extensions

### Shor-term (Weeks 2-4)
- [ ] Fine-tune CLIP on brand-specific data
- [ ] Add scene detection (instead of interval-based)
- [ ] Implement batch API endpoint

### Medium-term (Months 2-3)
- [ ] Train mood → KPI predictor
- [ ] Web dashboard for creative search
- [ ] Performance feedback integration

### Long-term (Months 3+)
- [ ] Human-in-the-loop system
- [ ] Organizational aesthetic knowledge base
- [ ] Real-time creative optimization

---

**Architecture Document Version:** 1.0  
**Last Updated:** February 2026  
**Status:** Complete ✓
