# Streamlit Web App - Setup & Run Guide

## 🚀 Quick Start

### 1. Install Streamlit

```powershell
pip install streamlit
```

### 2. Run the Web App

```powershell
streamlit run streamlit_app.py
```

This will open a browser window automatically at `http://localhost:8501`

---

## 📱 Features

### Interactive Pipeline
- 🎬 **Upload Video** - Drag & drop MP4/AVI files
- 🖼️ **Extract Frames** - Configure extraction interval
- 🧠 **Generate Embeddings** - With verification tests
- 📊 **Analyze Similarity** - Real-time visualizations
- 📈 **View Results** - Heatmaps, projections, reports

### Visualizations
- ✅ **Similarity Heatmap** - Frame-to-frame relationships
- ✅ **2D PCA Projection** - Mood/aesthetic clustering
- ✅ **Top-K Results** - Similar frames per query
- ✅ **Statistics Dashboard** - Real-time metrics

### Data Export
- 📥 Download similarity report (JSON)
- 📊 Query results with similarity scores
- 📁 Complete processing metadata

---

## 🎯 Usage Workflow

1. **Open the app**
   ```powershell
   streamlit run streamlit_app.py
   ```

2. **In the sidebar**, select each stage sequentially:
   - **Home** - Learn about the pipeline
   - **Upload Video** - Select your MP4/AVI
   - **Extract Frames** - Choose extraction interval
   - **Generate Embeddings** - Click extract
   - **Analyze Similarity** - Compute relationships
   - **Results** - View all visualizations

3. **Right panel shows**:
   - Live progress indicators
   - Metrics and statistics
   - All visualizations
   - Download options

---

## 🔧 Troubleshooting

### "streamlit not found"
```powershell
pip install streamlit
```

### Port 8501 already in use
```powershell
streamlit run streamlit_app.py --server.port 8502
```

### Large file uploads
The default limit is 200MB. To increase:
```powershell
streamlit run streamlit_app.py --logger.level=debug --client.maxUploadSize=500
```

---

## 📋 File Structure

```
vibe_project/
├── streamlit_app.py          ← Run this file
├── extract_frames.py         ← Used by app
├── embed_frames.py           ← Used by app
├── similarity_heatmap.py     ← Used by app
└── verification_tests.py     ← Used by app
```

---

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (FREE)
1. Push code to GitHub
2. Visit https://share.streamlit.io
3. Connect your GitHub repo
4. App runs automatically

### Option 2: Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

```powershell
docker build -t genta-app .
docker run -p 8501:8501 genta-app
```

### Option 3: Digital Ocean / AWS / Heroku
Many cloud providers support Streamlit directly.

---

## ✅ What You Get

A **fully functional web interface** for:
- ✅ Video upload & processing
- ✅ Real-time frame extraction
- ✅ CLIP embedding computation
- ✅ Interactive similarity analysis
- ✅ Live visualizations
- ✅ Results export
- ✅ Mobile-friendly interface

---

## 💡 Tips

1. **Best with smaller videos** (30 sec - 2 min)
2. **Uses CPU by default** (add GPU support if available)
3. **All processing in memory** (temp files auto-cleaned)
4. **Fully responsive design** (works on mobile)

---

## 🎓 Extending the App

Want to add more features?

```python
# Add custom metrics
st.metric("Custom Metric", value)

# Add more visualizations
st.line_chart(data)
st.map(data)

# Add export options
st.download_button(label, data, filename)
```

See Streamlit docs: https://docs.streamlit.io

---

**Ready to go! Run `streamlit run streamlit_app.py` now** 🚀
