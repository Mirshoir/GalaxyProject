#!/usr/bin/env python
"""
Galaxy Morphology Analyzer - Streamlit Frontend

Interactive web application for compression-based galaxy morphology analysis.
Imports functionality from modular source files.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import cv2
import tempfile
import json
import io
import base64

# Import modules from your project structure
try:
    from src.compression import NCDCalculator, CanonicalImageProcessor, Compressor
    from src.preprocess import GalaxyDataset, Preprocessor
    from src.nearest_neighbors import NCDNearestNeighbors
    from clustering import DistanceSpaceClustering, hierarchical_clustering
    from src.metrics import compute_rms_vs_k, average_rms_across_clusters
    from experiments.test_distances import test_synthetic_images, test_real_galaxies
    from experiments.rms_vs_k import run_rms_vs_k_analysis, visualize_rms_vs_k
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Please ensure the 'src' directory exists with all required modules.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Galaxy Morphology Analyzer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
def inject_custom_css():
    """Inject custom CSS styles."""
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
    }

    .cluster-0 { color: #FF6B6B; }
    .cluster-1 { color: #4ECDC4; }
    .cluster-2 { color: #FFD166; }
    .cluster-3 { color: #06D6A0; }
    .cluster-4 { color: #118AB2; }
    .cluster-5 { color: #EF476F; }

    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background: white;
        transition: all 0.3s ease;
    }

    .image-container:hover {
        border-color: #667eea;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}
    if 'distance_matrix' not in st.session_state:
        st.session_state.distance_matrix = None
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = {}
    if 'rms_results' not in st.session_state:
        st.session_state.rms_results = {}
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    if 'compression_algo' not in st.session_state:
        st.session_state.compression_algo = 'zlib'
    if 'ncd_calculator' not in st.session_state:
        st.session_state.ncd_calculator = NCDCalculator()


# Navigation
def create_sidebar():
    """Create navigation sidebar."""
    with st.sidebar:
        st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>
        🌌 Galaxy Analyzer
        </h1>
        """, unsafe_allow_html=True)

        page = st.selectbox(
            "Navigate to",
            [
                "🏠 Dashboard",
                "📊 Dataset Explorer",
                "🛠️ Preprocessing",
                "⚡ Compression Test",
                "📏 Distance Analysis",
                "🔍 Nearest Neighbors",
                "🗃️ Clustering",
                "📈 RMS Analysis",
                "⚙️ Settings"
            ]
        )

        st.markdown("---")

        # Quick stats
        if st.session_state.uploaded_files:
            st.markdown("### 📊 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Images", len(st.session_state.uploaded_files))
            with col2:
                if st.session_state.distance_matrix is not None:
                    n = len(st.session_state.distance_matrix)
                    st.metric("Distances", f"{n}×{n}")

        st.markdown("---")

        # Quick actions
        if st.button("🔄 Clear Cache", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

        return page


# Page 1: Dashboard
def page_dashboard():
    """Main dashboard page."""
    st.markdown("<h1 style='color: #667eea;'>🌌 Galaxy Morphology Dashboard</h1>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        ### Welcome to Galaxy Morphology Analyzer

        This application uses **Normalized Compression Distance (NCD)** to analyze 
        galaxy morphologies without requiring labels or trained models.

        **Key Features:**
        - 🚀 Model-free, unsupervised analysis
        - 🔬 Information-theoretic approach  
        - 📊 Interactive visualization
        - 🔍 Step-by-step debugging
        - 📈 Comprehensive metrics
        """)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
        <h3>📁 Images</h3>
        <h1>{len(st.session_state.uploaded_files)}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if st.session_state.distance_matrix is not None:
            n = len(st.session_state.distance_matrix)
            st.markdown(f"""
                <div class='metric-card'>
                <h3>📏 Distances</h3>
                <h1>{n}×{n}</h1>
                </div>
            """, unsafe_allow_html=True)

    # Quick start guide
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")

    steps = [
        ("1. 📊 Dataset Explorer", "Upload or load your galaxy images"),
        ("2. 🛠️ Preprocessing", "Configure image preprocessing parameters"),
        ("3. ⚡ Compression Test", "Test NCD on sample image pairs"),
        ("4. 📏 Distance Analysis", "Compute full distance matrix"),
        ("5. 🔍 Nearest Neighbors", "Find similar galaxies"),
        ("6. 🗃️ Clustering", "Cluster galaxies by morphology"),
        ("7. 📈 RMS Analysis", "Determine optimal cluster count")
    ]

    for step, desc in steps:
        with st.expander(f"{step}: {desc}", expanded=False):
            st.info(desc)

    # Recent activity
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("### 📅 Sample Images")

        # Display first 4 images
        sample_files = st.session_state.uploaded_files[:4]
        cols = st.columns(len(sample_files))

        for idx, (col, file_path) in enumerate(zip(cols, sample_files)):
            with col:
                try:
                    img = Image.open(file_path)
                    img.thumbnail((200, 200))
                    st.image(img, caption=f"Image {idx + 1}", use_column_width=True)
                except:
                    st.error("Error loading image")


# Page 2: Dataset Explorer
def page_dataset_explorer():
    """Dataset upload and management."""
    st.markdown("<h1 style='color: #667eea;'>📊 Dataset Explorer</h1>",
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📁 Manage", "👀 Preview"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # File uploader
            uploaded_files = st.file_uploader(
                "Choose galaxy images",
                type=['jpg', 'jpeg', 'png', 'tiff'],
                accept_multiple_files=True
            )

            if uploaded_files:
                temp_dir = tempfile.mkdtemp()
                new_files = []

                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    if file_path not in st.session_state.uploaded_files:
                        new_files.append(file_path)
                        st.session_state.uploaded_files.append(file_path)

                if new_files:
                    st.success(f"✅ Added {len(new_files)} new images")

            # Load from directory
            st.markdown("---")
            st.markdown("### Or Load from Directory")
            data_dir = st.text_input("Dataset Directory", "data/raw")

            if st.button("📂 Scan Directory"):
                if os.path.exists(data_dir):
                    dataset = GalaxyDataset(data_dir)
                    files = dataset.load_images("*.jpg")
                    st.session_state.uploaded_files = files
                    st.success(f"Found {len(files)} images")
                else:
                    st.error(f"Directory {data_dir} does not exist")

        with col2:
            # Dataset stats
            st.markdown("### 📊 Dataset Statistics")

            if st.session_state.uploaded_files:
                total_size = sum(os.path.getsize(f) for f in st.session_state.uploaded_files)

                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Total Images", len(st.session_state.uploaded_files))
                    st.metric("Total Size", f"{total_size / (1024 * 1024):.2f} MB")

                with col_stat2:
                    # File type distribution
                    extensions = {}
                    for file in st.session_state.uploaded_files:
                        ext = Path(file).suffix.lower()
                        extensions[ext] = extensions.get(ext, 0) + 1

                    if extensions:
                        fig = go.Figure(data=[go.Pie(
                            labels=list(extensions.keys()),
                            values=list(extensions.values()),
                            hole=0.3
                        )])
                        fig.update_layout(height=250, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if st.session_state.uploaded_files:
            # Display as dataframe
            data = []
            for idx, file_path in enumerate(st.session_state.uploaded_files):
                try:
                    img = Image.open(file_path)
                    data.append({
                        "ID": idx,
                        "Filename": Path(file_path).name,
                        "Size": img.size,
                        "Format": img.format,
                        "Path": file_path
                    })
                except:
                    data.append({
                        "ID": idx,
                        "Filename": Path(file_path).name,
                        "Size": "Error",
                        "Format": "Error",
                        "Path": file_path
                    })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # Batch operations
            st.markdown("---")
            selected_indices = st.multiselect(
                "Select images to remove",
                options=range(len(st.session_state.uploaded_files)),
                format_func=lambda x: Path(st.session_state.uploaded_files[x]).name
            )

            if selected_indices and st.button("Remove Selected"):
                # Remove in reverse to maintain indices
                for idx in sorted(selected_indices, reverse=True):
                    st.session_state.uploaded_files.pop(idx)
                st.rerun()

    with tab3:
        if st.session_state.uploaded_files:
            # Image grid preview
            n_cols = 4
            images_to_show = min(12, len(st.session_state.uploaded_files))

            for i in range(0, images_to_show, n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i + j
                    if idx < images_to_show:
                        with cols[j]:
                            try:
                                img = Image.open(st.session_state.uploaded_files[idx])
                                img.thumbnail((150, 150))
                                st.image(img, use_column_width=True)
                                st.caption(f"Image {idx}")
                            except:
                                st.error(f"Error loading")


# Page 3: Preprocessing
def page_preprocessing():
    """Image preprocessing configuration."""
    st.markdown("<h1 style='color: #667eea;'>🛠️ Preprocessing</h1>",
                unsafe_allow_html=True)

    if not st.session_state.uploaded_files:
        st.warning("Please upload images first!")
        return

    col_config, col_preview = st.columns(2)

    with col_config:
        st.markdown("### Configuration")

        # Configuration options
        image_size = st.slider("Image Size", 64, 512, 256, 32)
        normalize = st.checkbox("Normalize Intensities", True)
        apply_blur = st.checkbox("Apply Gaussian Blur", True)
        sigma = st.slider("Blur Sigma", 0.1, 5.0, 1.0, 0.1) if apply_blur else 0.0
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05) if normalize else None

        # Create processor
        processor = CanonicalImageProcessor(
            target_size=(image_size, image_size),
            normalize=normalize,
            apply_smoothing=apply_blur,
            sigma=sigma,
            threshold=threshold
        )

        # Test on selected image
        st.markdown("---")
        test_idx = st.selectbox(
            "Test on image",
            range(len(st.session_state.uploaded_files)),
            format_func=lambda x: Path(st.session_state.uploaded_files[x]).name
        )

        if st.button("Test Preprocessing"):
            try:
                # Process image
                processed_bytes = processor.process(st.session_state.uploaded_files[test_idx])
                processed_array = np.frombuffer(processed_bytes, dtype=np.uint8)
                processed_array = processed_array.reshape((image_size, image_size))
                processed_img = Image.fromarray(processed_array)

                # Store for preview
                st.session_state.preview_image = {
                    'original': Image.open(st.session_state.uploaded_files[test_idx]),
                    'processed': processed_img
                }
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with col_preview:
        st.markdown("### Preview")

        if 'preview_image' in st.session_state:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Original")
                st.image(st.session_state.preview_image['original'], use_column_width=True)

            with col2:
                st.markdown("#### Processed")
                st.image(st.session_state.preview_image['processed'], use_column_width=True)


# Page 4: Compression Test
def page_compression_test():
    """Test NCD compression."""
    st.markdown("<h1 style='color: #667eea;'>⚡ Compression Test</h1>",
                unsafe_allow_html=True)

    if len(st.session_state.uploaded_files) < 2:
        st.warning("Need at least 2 images for testing!")
        return

    tab1, tab2 = st.tabs(["🔍 Pair Test", "📊 Batch Test"])

    with tab1:
        col_sel, col_config = st.columns(2)

        with col_sel:
            img1_idx = st.selectbox(
                "Image 1",
                range(len(st.session_state.uploaded_files)),
                format_func=lambda x: Path(st.session_state.uploaded_files[x]).name
            )

            img2_idx = st.selectbox(
                "Image 2",
                range(len(st.session_state.uploaded_files)),
                format_func=lambda x: Path(st.session_state.uploaded_files[x]).name,
                index=1
            )

        with col_config:
            algorithm = st.selectbox(
                "Compression Algorithm",
                ['zlib', 'bz2', 'lzma'],
                index=0
            )

            test_same = st.checkbox("Test self-distance", True)

        # Display images
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            img1 = Image.open(st.session_state.uploaded_files[img1_idx])
            st.image(img1, caption="Image 1", use_column_width=True)

        with col_img2:
            img2 = Image.open(st.session_state.uploaded_files[img2_idx])
            st.image(img2, caption="Image 2", use_column_width=True)

        if st.button("🧪 Compute NCD"):
            try:
                calculator = NCDCalculator(Compressor(algorithm))

                results = {}

                # Self-distance test
                if test_same and img1_idx == img2_idx:
                    dist, comp_size = calculator.compute_self_distance(
                        st.session_state.uploaded_files[img1_idx]
                    )
                    results['self_distance'] = dist
                    results['compression_size'] = comp_size

                # Pair distance
                if img1_idx != img2_idx:
                    dist = calculator.ncd(
                        st.session_state.uploaded_files[img1_idx],
                        st.session_state.uploaded_files[img2_idx]
                    )
                    results['pair_distance'] = dist

                # Display results
                st.markdown("### 📊 Results")

                if 'self_distance' in results:
                    st.metric("Self Distance", f"{results['self_distance']:.6f}")
                    st.caption("Should be close to 0")

                if 'pair_distance' in results:
                    st.metric("Pair Distance", f"{results['pair_distance']:.4f}")

                    # Interpretation
                    if results['pair_distance'] < 0.1:
                        st.success("Images are very similar")
                    elif results['pair_distance'] < 0.3:
                        st.info("Images are somewhat similar")
                    elif results['pair_distance'] < 0.5:
                        st.warning("Images are different")
                    else:
                        st.error("Images are very different")

            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        if len(st.session_state.uploaded_files) >= 3:
            n_images = st.slider("Number of images", 3,
                                 min(10, len(st.session_state.uploaded_files)), 5)

            if st.button("Run Batch Test"):
                with st.spinner("Computing distances..."):
                    try:
                        calculator = NCDCalculator()
                        sample_files = st.session_state.uploaded_files[:n_images]

                        # Simple progress bar
                        progress_bar = st.progress(0)

                        # Compute distances
                        distances = np.zeros((n_images, n_images))
                        for i in range(n_images):
                            for j in range(i, n_images):
                                if i == j:
                                    distances[i, j] = 0.0
                                else:
                                    dist = calculator.ncd(sample_files[i], sample_files[j])
                                    distances[i, j] = dist
                                    distances[j, i] = dist

                            progress_bar.progress((i + 1) / n_images)

                        progress_bar.empty()

                        # Store results
                        st.session_state.batch_test_distances = distances

                        # Display heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=distances,
                            colorscale='Viridis'
                        ))

                        fig.update_layout(
                            title="Distance Matrix",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")


# Page 5: Distance Analysis
def page_distance_analysis():
    """Full distance matrix computation."""
    st.markdown("<h1 style='color: #667eea;'>📏 Distance Analysis</h1>",
                unsafe_allow_html=True)

    if not st.session_state.uploaded_files:
        st.warning("Please upload images first!")
        return

    st.markdown("### Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_images = st.slider(
            "Number of images",
            2,
            min(50, len(st.session_state.uploaded_files)),
            min(10, len(st.session_state.uploaded_files))
        )

    with col2:
        algorithm = st.selectbox(
            "Algorithm",
            ['zlib', 'bz2', 'lzma'],
            index=0
        )

    with col3:
        use_cache = st.checkbox("Use cached results", True)

    if st.button("🚀 Compute Distance Matrix", type="primary"):
        with st.spinner(f"Computing {n_images}×{n_images} distances..."):
            try:
                calculator = NCDCalculator(Compressor(algorithm))
                selected_files = st.session_state.uploaded_files[:n_images]

                # Initialize progress
                progress_text = st.empty()
                progress_bar = st.progress(0)

                # Compute distances
                distances = np.zeros((n_images, n_images))
                total_pairs = (n_images * (n_images - 1)) // 2
                pair_count = 0

                for i in range(n_images):
                    for j in range(i + 1, n_images):
                        dist = calculator.ncd(selected_files[i], selected_files[j])
                        distances[i, j] = dist
                        distances[j, i] = dist

                        pair_count += 1
                        progress_bar.progress(pair_count / total_pairs)
                        progress_text.text(f"Processed {pair_count}/{total_pairs} pairs")

                progress_bar.empty()
                progress_text.empty()

                # Store results
                st.session_state.distance_matrix = distances
                st.session_state.selected_files = selected_files

                st.success(f"✅ Computed distance matrix!")

            except Exception as e:
                st.error(f"Error: {e}")

    # Display results if available
    if st.session_state.distance_matrix is not None:
        distances = st.session_state.distance_matrix

        st.markdown("---")
        st.markdown("### 📊 Results")

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=distances,
            colorscale='RdBu_r',
            zmin=0,
            zmax=1
        ))

        fig.update_layout(
            title="Distance Matrix Heatmap",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        triu_indices = np.triu_indices_from(distances, k=1)
        flat_distances = distances[triu_indices]

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        with col_stat1:
            st.metric("Mean", f"{np.mean(flat_distances):.4f}")

        with col_stat2:
            st.metric("Std Dev", f"{np.std(flat_distances):.4f}")

        with col_stat3:
            st.metric("Min", f"{np.min(flat_distances):.4f}")

        with col_stat4:
            st.metric("Max", f"{np.max(flat_distances):.4f}")

        # Distribution plot
        fig_dist, ax = plt.subplots(figsize=(8, 4))
        ax.hist(flat_distances, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.set_title('Distance Distribution')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig_dist)


# Page 6: Nearest Neighbors
def page_nearest_neighbors():
    """Find similar galaxies."""
    st.markdown("<h1 style='color: #667eea;'>🔍 Nearest Neighbors</h1>",
                unsafe_allow_html=True)

    if st.session_state.distance_matrix is None:
        st.warning("Please compute distance matrix first!")
        return

    distances = st.session_state.distance_matrix
    selected_files = st.session_state.get('selected_files', [])

    col_query, col_config = st.columns(2)

    with col_query:
        query_idx = st.selectbox(
            "Query Image",
            range(len(selected_files)),
            format_func=lambda x: Path(selected_files[x]).name
        )

        # Display query image
        query_img = Image.open(selected_files[query_idx])
        st.image(query_img, caption="Query Image", use_column_width=True)

    with col_config:
        k = st.slider("Number of neighbors", 1, min(10, len(selected_files) - 1), 5)
        max_distance = st.slider("Max distance", 0.0, 1.0, 0.5, 0.05)

    if st.button("🔍 Find Neighbors"):
        # Get distances for query
        query_distances = distances[query_idx]

        # Sort and filter
        sorted_indices = np.argsort(query_distances)
        neighbors = []

        for idx in sorted_indices:
            if idx == query_idx:
                continue
            if query_distances[idx] > max_distance:
                continue
            if len(neighbors) >= k:
                break

            neighbors.append({
                'index': idx,
                'distance': query_distances[idx],
                'filename': Path(selected_files[idx]).name
            })

        # Display results
        if neighbors:
            st.markdown("### 📊 Nearest Neighbors")

            # Create results table
            results_data = []
            for rank, neighbor in enumerate(neighbors):
                results_data.append({
                    'Rank': rank + 1,
                    'Image': neighbor['filename'],
                    'Distance': f"{neighbor['distance']:.4f}",
                    'Similarity': f"{(1 - neighbor['distance']):.1%}"
                })

            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)

            # Visualize neighbors
            st.markdown("#### Neighbor Visualization")

            n_cols = min(4, len(neighbors) + 1)
            cols = st.columns(n_cols)

            # Query image
            with cols[0]:
                st.image(query_img, caption="Query", use_column_width=True)

            # Neighbors
            for i, neighbor in enumerate(neighbors[:n_cols - 1]):
                with cols[i + 1]:
                    neighbor_img = Image.open(selected_files[neighbor['index']])
                    st.image(neighbor_img,
                             caption=f"Rank {i + 1}\nDist: {neighbor['distance']:.3f}",
                             use_column_width=True)
        else:
            st.warning("No neighbors found within distance threshold.")


# Page 7: Clustering
def page_clustering():
    """Cluster galaxies by morphology."""
    st.markdown("<h1 style='color: #667eea;'>🗃️ Clustering</h1>",
                unsafe_allow_html=True)

    if st.session_state.distance_matrix is None:
        st.warning("Please compute distance matrix first!")
        return

    distances = st.session_state.distance_matrix

    col_config, col_results = st.columns(2)

    with col_config:
        st.markdown("### Configuration")

        algorithm = st.selectbox(
            "Clustering Algorithm",
            ["K-Medoids", "Hierarchical"],
            index=0
        )

        if algorithm == "K-Medoids":
            n_clusters = st.slider("Number of clusters", 2, min(10, len(distances)), 3)
        else:
            n_clusters = st.slider("Number of clusters", 2, min(10, len(distances)), 3)
            linkage_method = st.selectbox("Linkage method",
                                          ['ward', 'complete', 'average', 'single'])

        if st.button("🔮 Cluster Images"):
            with st.spinner("Clustering..."):
                try:
                    if algorithm == "K-Medoids":
                        from sklearn_extra.cluster import KMedoids
                        clusterer = KMedoids(n_clusters=n_clusters,
                                             metric='precomputed',
                                             random_state=42)
                        labels = clusterer.fit_predict(distances)
                    else:
                        labels = hierarchical_clustering(distances, n_clusters, linkage_method)

                    # Store results
                    st.session_state.clustering_results = {
                        'algorithm': algorithm,
                        'labels': labels,
                        'n_clusters': n_clusters
                    }

                    st.success(f"✅ Found {n_clusters} clusters!")

                except Exception as e:
                    st.error(f"Error: {e}")

    with col_results:
        if 'clustering_results' in st.session_state:
            results = st.session_state.clustering_results
            labels = results['labels']

            st.markdown("### 📊 Cluster Results")

            # Cluster sizes
            unique_labels = np.unique(labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]

            fig = go.Figure(data=[go.Bar(
                x=[f'Cluster {i}' for i in unique_labels],
                y=cluster_sizes,
                marker_color=px.colors.qualitative.Set3[:len(unique_labels)]
            )])

            fig.update_layout(
                title="Cluster Sizes",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show cluster members
            st.markdown("#### Cluster Members")

            for cluster_id in unique_labels:
                with st.expander(f"Cluster {cluster_id} ({cluster_sizes[cluster_id]} images)"):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    for idx in cluster_indices[:5]:  # Show first 5
                        st.text(f"Image {idx}")

                    if len(cluster_indices) > 5:
                        st.caption(f"... and {len(cluster_indices) - 5} more")


# Page 8: RMS Analysis
def page_rms_analysis():
    """RMS vs k analysis."""
    st.markdown("<h1 style='color: #667eea;'>📈 RMS Analysis</h1>",
                unsafe_allow_html=True)

    if st.session_state.distance_matrix is None:
        st.warning("Please compute distance matrix first!")
        return

    distances = st.session_state.distance_matrix

    col_config, col_results = st.columns(2)

    with col_config:
        st.markdown("### Configuration")

        min_k = st.slider("Minimum k", 2, 10, 2)
        max_k = st.slider("Maximum k", 3, 15, 8)

        if min_k >= max_k:
            st.error("Minimum k must be less than maximum k")
            return

        n_runs = st.slider("Runs per k", 1, 5, 3)

        if st.button("📊 Run RMS Analysis"):
            with st.spinner("Running RMS analysis..."):
                try:
                    k_values, mean_rms, std_rms = compute_rms_vs_k(
                        distances,
                        list(range(min_k, max_k + 1)),
                        n_runs,
                        42
                    )

                    # Find elbow point
                    from kneed import KneeLocator
                    kl = KneeLocator(k_values, mean_rms, curve='convex', direction='decreasing')
                    elbow_k = kl.elbow

                    # Store results
                    st.session_state.rms_results = {
                        'k_values': k_values.tolist(),
                        'mean_rms': mean_rms.tolist(),
                        'std_rms': std_rms.tolist(),
                        'elbow_k': elbow_k
                    }

                    st.success(f"✅ Analysis complete! Suggested k = {elbow_k}")

                except Exception as e:
                    st.error(f"Error: {e}")

    with col_results:
        if 'rms_results' in st.session_state:
            results = st.session_state.rms_results

            # Plot RMS vs k
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=results['k_values'],
                y=results['mean_rms'],
                mode='lines+markers',
                name='RMS',
                error_y=dict(
                    type='data',
                    array=results['std_rms'],
                    visible=True
                )
            ))

            # Highlight elbow point
            if results['elbow_k']:
                elbow_idx = results['k_values'].index(results['elbow_k'])
                fig.add_trace(go.Scatter(
                    x=[results['elbow_k']],
                    y=[results['mean_rms'][elbow_idx]],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name=f'Elbow (k={results["elbow_k"]})'
                ))

            fig.update_layout(
                title="RMS vs Number of Clusters",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Average RMS Distance",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show table
            st.markdown("#### Results Table")

            data = []
            for k, rms, std in zip(results['k_values'],
                                   results['mean_rms'],
                                   results['std_rms']):
                data.append({
                    'k': k,
                    'RMS': f"{rms:.4f}",
                    'Std Dev': f"{std:.4f}",
                    'Elbow': '⭐' if k == results['elbow_k'] else ''
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)


# Page 9: Settings
def page_settings():
    """Application settings."""
    st.markdown("<h1 style='color: #667eea;'>⚙️ Settings</h1>",
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["General", "About"])

    with tab1:
        st.markdown("### General Settings")

        # Data directory
        data_dir = st.text_input("Default Data Directory", "data/raw")

        # Image processing defaults
        st.markdown("#### Image Processing Defaults")
        default_size = st.slider("Default Image Size", 64, 512, 256, 32)
        default_normalize = st.checkbox("Default Normalization", True)

        # Compression defaults
        st.markdown("#### Compression Defaults")
        default_algo = st.selectbox(
            "Default Algorithm",
            ['zlib', 'bz2', 'lzma'],
            index=0
        )

        if st.button("💾 Save Settings"):
            st.success("Settings saved! (Note: In this demo, settings are session-only)")

    with tab2:
        st.markdown("### About")

        st.markdown("""
        #### 🌌 Galaxy Morphology Analyzer

        Version 1.0.0

        A model-free, unsupervised approach to galaxy morphology classification 
        using Normalized Compression Distance (NCD).

        **Features:**
        - No training data or labels required
        - Information-theoretic foundation
        - Step-by-step analysis
        - Comprehensive debugging
        - Advanced visualization

        **Methodology:**
        1. Convert images to canonical representation
        2. Compute compression complexity
        3. Calculate pairwise NCD distances
        4. Analyze clustering behavior
        5. Determine optimal cluster count

        **License:** MIT Open Source
        """)


# Main app
def main():
    """Main application."""
    # Inject CSS
    inject_custom_css()

    # Initialize session state
    init_session_state()

    # Create navigation
    page = create_sidebar()

    # Route to selected page
    pages = {
        "🏠 Dashboard": page_dashboard,
        "📊 Dataset Explorer": page_dataset_explorer,
        "🛠️ Preprocessing": page_preprocessing,
        "⚡ Compression Test": page_compression_test,
        "📏 Distance Analysis": page_distance_analysis,
        "🔍 Nearest Neighbors": page_nearest_neighbors,
        "🗃️ Clustering": page_clustering,
        "📈 RMS Analysis": page_rms_analysis,
        "⚙️ Settings": page_settings
    }

    # Execute selected page
    if page in pages:
        pages[page]()
    else:
        st.error(f"Page '{page}' not found")


if __name__ == "__main__":
    # Import matplotlib here to avoid warning
    import matplotlib.pyplot as plt

    plt.style.use('default')

    # Run the app
    main()