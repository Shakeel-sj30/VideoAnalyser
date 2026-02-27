"""
GenTA Affective Computing Pipeline - Streamlit Web App
Interactive web interface for mood/style similarity analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Import pipeline modules
from extract_frames import FrameExtractor
from embed_frames import FrameEmbedder, save_embeddings
from similarity_heatmap import VibeSimilarityAnalyzer

# Page config
st.set_page_config(
    page_title="GenTA Affective Computing Pipeline",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .header {
        text-align: center;
        padding: 20px;
    }
    .success-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'frames_extracted': False,
        'embeddings_computed': False,
        'analysis_complete': False,
        'frames_dir': None,
        'embeddings': None,
        'metadata': None,
        'analyzer': None
    }

# Sidebar
with st.sidebar:
    st.title("🎨 GenTA Pipeline")
    st.write("Affective Computing for Art & Marketing Visuals")
    
    pipeline_step = st.radio(
        "Select Pipeline Stage:",
        ["Home", "Upload Video", "Extract Frames", "Generate Embeddings", "Analyze Similarity", "Results"]
    )
    
    st.divider()
    st.write("### ℹ️ About")
    st.write("""
    This pipeline analyzes the "vibe" (mood/style) of video frames using AI embeddings.
    
    **Steps:**
    1. Upload a video (MP4, AVI)
    2. Extract frames automatically
    3. Generate CLIP embeddings
    4. Compute frame similarity
    5. Visualize & analyze results
    """)

# Main Content
st.title("🎨 GenTA Affective Computing Pipeline")
st.write("Understanding the 'vibe' of contemporary art and marketing visuals through AI")

if pipeline_step == "Home":
    st.markdown("""
    ## Welcome to GenTA's Affective Computing Engine
    
    This interactive application helps you understand and quantify the "vibe" (emotional resonance, 
    aesthetic coherence) of video content through AI-powered embeddings.
    
    ### How It Works
    
    **Step 1: Upload Video** 📹
    - Upload a short marketing or art video (MP4/AVI)
    - Length: 30 seconds to 2 minutes optimal
    
    **Step 2: Extract Frames** 🖼️
    - Automatically extracts representative frames (1 per second)
    - Creates organized metadata for tracking
    
    **Step 3: Generate Embeddings** 🧠
    - Uses CLIP model to convert frames to mood/style vectors
    - 512-dimensional semantic representation
    - Includes verification tests
    
    **Step 4: Analyze Similarity** 📊
    - Computes pairwise frame similarity
    - Finds frames with similar "vibe"
    - Generates visualizations
    
    ### Key Features
    
    ✅ **Verification-First** - All outputs validated for correctness  
    ✅ **Production-Ready** - Robust error handling and logging  
    ✅ **Interpretable** - Visualizations show mood/aesthetic relationships  
    ✅ **Scalable** - Architecture ready for KPI integration  
    
    ### Getting Started
    
    1. Click **"Upload Video"** in the sidebar
    2. Select your MP4/AVI file
    3. Click through each pipeline stage
    4. View results and visualizations
    
    ---
    
    **GenTA Vision:** Make contemporary art and marketing creatives emotionally accessible 
    through automated aesthetic understanding.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pipeline Stages", 4, "Extract → Embed → Analyze → Visualize")
    with col2:
        st.metric("Embedding Dimension", 512, "CLIP-ViT-B32")
    with col3:
        st.metric("Processing Speed", "~100ms", "per frame (CPU)")

elif pipeline_step == "Upload Video":
    st.header("📹 Upload Your Video")
    
    st.write("""
    Upload a short marketing or art video to begin analysis.
    
    **Recommended specs:**
    - Format: MP4 or AVI
    - Duration: 30 seconds to 2 minutes
    - Resolution: 720p or higher
    """)
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Save uploaded file to temp directory
        temp_dir = tempfile.mkdtemp()
        video_path = Path(temp_dir) / uploaded_file.name
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.pipeline_state['video_path'] = str(video_path)
        st.session_state.pipeline_state['video_name'] = uploaded_file.name
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📁 Temp Path: {video_path}")
        with col2:
            st.info(f"📊 Size: {uploaded_file.size / (1024*1024):.2f} MB")

elif pipeline_step == "Extract Frames":
    st.header("🖼️ Extract Frames from Video")
    
    if 'video_path' not in st.session_state.pipeline_state:
        st.warning("⚠️ Please upload a video first in the 'Upload Video' section")
    else:
        st.write(f"Video: {st.session_state.pipeline_state['video_name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            interval = st.slider("Frame interval (seconds)", 0.5, 5.0, 1.0)
        with col2:
            st.info(f"Will extract ~{int(60 / interval)} frames per minute of video")
        
        if st.button("🚀 Extract Frames", use_container_width=True):
            with st.spinner("Extracting frames..."):
                try:
                    # Create temp directories
                    temp_root = Path(tempfile.mkdtemp())
                    frames_dir = temp_root / "frames"
                    frames_dir.mkdir()
                    
                    # Extract frames
                    extractor = FrameExtractor(output_dir=str(frames_dir))
                    saved, metadata = extractor.extract_frames(
                        st.session_state.pipeline_state['video_path'],
                        frame_interval_seconds=interval
                    )
                    
                    # Save metadata
                    extractor.save_metadata(str(frames_dir / "metadata.json"))
                    
                    # Update session state
                    st.session_state.pipeline_state['frames_extracted'] = True
                    st.session_state.pipeline_state['frames_dir'] = str(frames_dir)
                    st.session_state.pipeline_state['frame_metadata'] = {
                        'count': saved,
                        'interval': interval
                    }
                    
                    st.success(f"✅ Extracted {saved} frames!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Frames Extracted", saved)
                    with col2:
                        st.metric("Interval", f"{interval}s")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif pipeline_step == "Generate Embeddings":
    st.header("🧠 Generate CLIP Embeddings")
    
    if not st.session_state.pipeline_state.get('frames_extracted'):
        st.warning("⚠️ Please extract frames first")
    else:
        frames_dir = st.session_state.pipeline_state['frames_dir']
        frame_count = st.session_state.pipeline_state['frame_metadata']['count']
        
        st.info(f"📂 Processing {frame_count} frames from {frames_dir}")
        
        if st.button("🚀 Generate Embeddings", use_container_width=True):
            with st.spinner("Loading CLIP model and computing embeddings..."):
                try:
                    # Initialize embedder
                    embedder = FrameEmbedder()
                    
                    # Compute embeddings
                    embeddings, metadata = embedder.embed_frames(
                        frame_dir=frames_dir,
                        metadata_file=str(Path(frames_dir) / "metadata.json")
                    )
                    
                    # Save embeddings
                    embeddings_dir = Path(frames_dir).parent / "embeddings"
                    embeddings_dir.mkdir(exist_ok=True)
                    save_embeddings(embeddings, metadata, output_dir=str(embeddings_dir))
                    
                    # Update session state
                    st.session_state.pipeline_state['embeddings_computed'] = True
                    st.session_state.pipeline_state['embeddings'] = embeddings
                    st.session_state.pipeline_state['metadata'] = metadata
                    st.session_state.pipeline_state['embeddings_dir'] = str(embeddings_dir)
                    
                    st.success("✅ Embeddings generated successfully!")
                    
                    # Show verification results
                    st.subheader("Verification Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Shape", f"{embeddings.shape[0]}×{embeddings.shape[1]}")
                    with col2:
                        st.metric("NaN Count", "0" if not np.isnan(embeddings).any() else "❌")
                    with col3:
                        st.metric("Inf Count", "0" if not np.isinf(embeddings).any() else "❌")
                    
                    # Show statistics
                    st.subheader("Embedding Statistics")
                    norms = np.linalg.norm(embeddings, axis=1)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("L2 Norm (Mean)", f"{norms.mean():.4f}")
                    with col2:
                        st.metric("L2 Norm (Min)", f"{norms.min():.4f}")
                    with col3:
                        st.metric("L2 Norm (Max)", f"{norms.max():.4f}")
                    with col4:
                        st.metric("Embedding Dim", embeddings.shape[1])
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif pipeline_step == "Analyze Similarity":
    st.header("📊 Analyze Frame Similarity")
    
    if not st.session_state.pipeline_state.get('embeddings_computed'):
        st.warning("⚠️ Please generate embeddings first")
    else:
        embeddings = st.session_state.pipeline_state['embeddings']
        embeddings_dir = st.session_state.pipeline_state['embeddings_dir']
        
        if st.button("🚀 Analyze Similarity", use_container_width=True):
            with st.spinner("Computing similarity and generating visualizations..."):
                try:
                    # Initialize analyzer
                    analyzer = VibeSimilarityAnalyzer(
                        embeddings_path=str(Path(embeddings_dir) / "frame_embeddings.npy"),
                        metadata_path=str(Path(embeddings_dir) / "embeddings_metadata.json")
                    )
                    
                    st.session_state.pipeline_state['analyzer'] = analyzer
                    st.session_state.pipeline_state['analysis_complete'] = True
                    
                    # Print statistics
                    st.subheader("Similarity Statistics")
                    analyzer.print_statistics()
                    
                    # Generate visualizations
                    st.subheader("Generating Visualizations...")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Similarity Heatmap**")
                        fig_heat, ax = plt.subplots(figsize=(8, 7))
                        sns.heatmap(analyzer.similarity_matrix, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Similarity"})
                        ax.set_title("Frame-to-Frame Vibe Similarity")
                        ax.set_xlabel("Frame Index")
                        ax.set_ylabel("Frame Index")
                        st.pyplot(fig_heat)
                    
                    with col2:
                        st.write("**2D PCA Projection**")
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        emb_2d = pca.fit_transform(embeddings)
                        
                        fig_pca, ax = plt.subplots(figsize=(8, 7))
                        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                           c=range(len(embeddings)), cmap='hsv', s=100, alpha=0.7)
                        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                        ax.set_title("2D Embedding Projection")
                        plt.colorbar(scatter, ax=ax, label="Frame Index")
                        st.pyplot(fig_pca)
                    
                    st.success("✅ Analysis Complete!")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif pipeline_step == "Results":
    st.header("📈 Results & Interpretation")
    
    if not st.session_state.pipeline_state.get('analysis_complete'):
        st.warning("⚠️ Please complete the analysis first")
    else:
        analyzer = st.session_state.pipeline_state['analyzer']
        embeddings = st.session_state.pipeline_state['embeddings']
        
        # Similarity statistics
        st.subheader("📊 Vibe Similarity Metrics")
        
        upper_tri_indices = np.triu_indices_from(analyzer.similarity_matrix, k=1)
        similarities = analyzer.similarity_matrix[upper_tri_indices]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Mean", f"{similarities.mean():.4f}")
        with col2:
            st.metric("Std Dev", f"{similarities.std():.4f}")
        with col3:
            st.metric("Min", f"{similarities.min():.4f}")
        with col4:
            st.metric("Max", f"{similarities.max():.4f}")
        with col5:
            st.metric("Median", f"{np.median(similarities):.4f}")
        
        # Top-K Similar Frames
        st.subheader("🎯 Top-5 Similar Frames (Query-Based Retrieval)")
        
        query_indices = [0, len(embeddings)//2, len(embeddings)-1]
        
        for q_idx in query_indices:
            with st.expander(f"Query Frame {q_idx}", expanded=False):
                top_k = analyzer.get_top_k_similar(q_idx, k=5)
                
                results_data = {
                    'Rank': list(range(1, 6)),
                    'Frame Index': [idx for idx, _, _ in top_k],
                    'Similarity': [f"{sim:.4f}" for _, sim, _ in top_k],
                    'Filename': [fname for _, _, fname in top_k]
                }
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 4))
                sims = [sim for _, sim, _ in top_k]
                indices = [f"Frame {idx}" for idx, _, _ in top_k]
                bars = ax.barh(indices, sims, color='steelblue')
                ax.set_xlim(0, 1.0)
                ax.set_xlabel("Similarity Score")
                ax.set_title(f"Top-5 Similar Frames to Query {q_idx}")
                
                for bar, sim in zip(bars, sims):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f'{sim:.3f}', ha='left', va='center', fontweight='bold')
                
                st.pyplot(fig)
        
        # Interpretation
        st.subheader("💡 GenTA Interpretation")
        
        if similarities.mean() > 0.8:
            st.success("""
            **High Coherence:** Your video has a consistent mood/aesthetic throughout.
            - Frames are visually/emotionally similar
            - Strong brand/style consistency
            - Good for unified marketing message
            """)
        else:
            st.info("""
            **Varied Aesthetic:** Your video has diverse mood/aesthetic elements.
            - Different emotional tones across frames
            - Good for dynamic storytelling
            - Captures viewer attention through variation
            """)
        
        # Export Results
        st.subheader("📥 Download Results")
        
        # Generate report
        report = analyzer.generate_report(query_indices=query_indices, k=5)
        report_json = json.dumps(report, indent=2)
        
        st.download_button(
            label="📄 Download JSON Report",
            data=report_json,
            file_name="vibe_similarity_report.json",
            mime="application/json"
        )

# Footer
st.divider()
st.markdown("""
---
**GenTA Affective Computing Pipeline** | Built with Streamlit | 🎨 Understanding the "vibe" of visual content
""")
