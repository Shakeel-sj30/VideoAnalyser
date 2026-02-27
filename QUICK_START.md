# QUICK START GUIDE
## GenTA Affective Computing Pipeline

---

## ⚡ 5-Minute Setup

### 1. Install & Activate

```bash
cd vibe_project
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Videos

```bash
# Download 2-3 short videos from:
# - Pexels: https://www.pexels.com/videos/
# - Pixabay: https://pixabay.com/videos/
# - Save as .mp4 files in ./videos/ folder

python setup_videos.py  # Shows instructions
```

### 3. Run Pipeline

```bash
python extract_frames.py      # Extract frames
python embed_frames.py        # Generate embeddings
python similarity_heatmap.py  # Analyze & visualize
```

### 4. View Results

```
outputs/
├── similarity_heatmap.png          # Heat map of frame relationships
├── embedding_projection_2d.png     # PCA visualization
└── query_results_bars.png          # Top-5 similar frames per query

embeddings/
├── similarity_report.json          # Detailed results
└── frame_embeddings.npy            # Raw embedding vectors
```

---

## 📚 Documentation

- **README.md** - Full documentation, API reference, troubleshooting
- **GenTA_Affective_Computing_Pipeline.ipynb** - Interactive notebook with explanations
- **AI_TOOL_USAGE_AND_VERIFICATION.md** - How AI tools were used and verified
- **PROJECT_SUMMARY.md** - Architecture and GenTA context

---

## 🎯 What This Does

**Input:** 2-3 short marketing/art videos  
**Process:** Extracts frames → computes mood embeddings → finds similar vibes  
**Output:** Visualizations showing which frames have similar aesthetic/emotional properties

**GenTA Application:** Foundation for affective computing engine that understands the "feel" of marketing creatives and art content.

---

## 🏥 Verify Installation

```bash
python -c "import torch, transformers; print('✓ All packages imported successfully')"
```

---

## ⚠️ Common Issues

**"ModuleNotFoundError: No module named 'torch'"**
```bash
pip install -r requirements.txt
```

**"No video files found"**
- Add .mp4 files to `./videos/` folder
- Run: `python setup_videos.py` for download instructions

**"Out of memory"**
- Use CPU: Change device in `embed_frames.py` to `"cpu"`
- Or reduce batch size: `batch_size=16` on GPU

**More help:** See README.md Troubleshooting section

---

## 📊 Pipeline Output Example

```
✓ Frame Extraction: 120 frames extracted
✓ Embedding Generation: 120 embeddings computed (512-dim each)
✓ Verification Tests: 6/6 passed
  - Shape ✓
  - NaN/Inf ✓
  - Normalization ✓
  - Self-similarity ✓
  - Diversity ✓
  - Duplicate detection ✓

✓ Similarity Analysis: 
  - Mean similarity: 0.524
  - Std dev: 0.182
  - Query 0: 5 similar frames found
  - Query 60: 5 similar frames found
  - Query 120: 5 similar frames found

✓ Visualizations: 3 figures generated
✓ Reports: JSON exports completed
```

---

## 🤔 How It Works (30-Second Version)

1. **Frame Extraction** → OpenCV extracts 1 frame per second from videos
2. **CLIP Embeddings** → HuggingFace CLIP converts images to 512-dim vectors representing mood/style
3. **Similarity** → Calculates cosine similarity between all frame vectors (which frames have similar "vibe")
4. **Visualization** → Heatmaps & charts show frame relationships

**Why CLIP?** It understands semantic similarity (mood, style, aesthetic) beyond pixel-level features.

---

## 🚀 Next Level: Interactive Exploration

View the Jupyter notebook for detailed explanations and visualizations:

```bash
jupyter notebook GenTA_Affective_Computing_Pipeline.ipynb
```

This notebook includes:
- Full walkthrough of each pipeline stage
- Inline visualizations
- Verification tests with results
- GenTA affective computing context
- Next steps for GACS integration

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Video files in `./videos/` folder (2-3 videos, MP4 format)
- [ ] Ran `python extract_frames.py` (no errors)
- [ ] Ran `python embed_frames.py` (embeddings generated)
- [ ] Ran `python similarity_heatmap.py` (visualizations created)
- [ ] Checked outputs in `./outputs/` and `./embeddings/`

---

## 📞 Getting Help

1. **Is something broken?** → See README.md Troubleshooting
2. **How does it work?** → Open Jupyter notebook
3. **How was it built?** → See AI_TOOL_USAGE_AND_VERIFICATION.md
4. **What's next?** → See PROJECT_SUMMARY.md Next Steps

---

## 🎯 Your Mission (If You Choose to Accept)

**Tomorrow:** Extend this to connect embeddings →KPI predictions (CTR/ROAS)  
**Next Week:** Fine-tune CLIP on your brand data  
**Next Month:** Deploy as real-time creative scoring API

This prototype is your foundation. Build the GACS engine on top. ✓

---

**Happy computing! 🚀**
