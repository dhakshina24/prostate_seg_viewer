"""Streamlit app for visualizing prostate cancer segmenatation results from PiCAI nnUNet Docker container"""

# Imports 
import json
import os
from pathlib import Path
import numpy as np
import streamlit as st

from utils.data_loader import load_mri, load_mask
from overlay import transparent_overlay
from config import SEG_PATH, SCORE_PATH, INPUT_DIR, OUTPUT_DIR
from inference import run_inference

st.set_page_config(page_title="Prostate Cancer Segmentation Viewer", layout="wide")


@st.cache_data
def load_all_modalities():
    """Load T2, ADC and HBV MRI modalities and their metadata"""

    input_dir = Path(INPUT_DIR)

    def find_file(pattern: str) -> str:
        """Return matching file path for a given pattern"""
        matches = list(input_dir.glob(pattern))
        if not matches:
            st.error(f"No match found for pattern: {pattern}")
            st.stop()
        return matches[0].as_posix()
    
    t2_path = find_file("*/transverse-t2*/*.mha")
    adc_path = find_file("*/transverse-adc*/*.mha")
    hbv_path = find_file("*/transverse-hbv*/*.mha")

    t2_data, t2 = load_mri(t2_path)
    adc_data, adc = load_mri(adc_path)
    hbv_data, hbv = load_mri(hbv_path)

    return { 
        "t2": (t2_data, t2),
        "adc": (adc_data, adc),
        "hbv": (hbv_data, hbv),
    }

def load_segmentation():
    """Load Segmentation mask and metadata"""
    mask_data, mask = load_mask(SEG_PATH)
    return mask_data, mask

def display_results(t2, mask, mask_data):

    # Center image for display
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Slider for segmentation slice selection
        slice_result_idx = st.sidebar.slider("Segmentation Slice", 0, mask.shape[0] - 1,  mask.shape[0] // 2)

        # Slider for opacity selection
        opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.4, step=0.1)

        # Display Image and colorbar
        fig = transparent_overlay(t2[slice_result_idx, :, :], mask[slice_result_idx, :, :], alpha=opacity)
        st.pyplot(fig)

        # Display metadata in sidebar
        with st.sidebar.expander("T2 segmentation mask", expanded=False):
            for key, val in mask_data.items():
                st.write(f"**{key}:** {val}")

def load_inference():
    """Run or load inference results, returning True if successful."""
    # Load existing results from previous inference
    if os.path.exists(SEG_PATH) and os.path.exists(SCORE_PATH):
        st.sidebar.success("Loading cached results.")
        return True

    # Run Inference using Docker Image
    with st.spinner("Running inference..."):
        finished = False
        for logs, finished in run_inference(INPUT_DIR, OUTPUT_DIR):
            print(logs)
    
    if finished:
        st.success("Inference complete")
        return True
    else:
        st.error("Inference failed to complete")
        return False

# Flag to track inference execution
if "inference" not in st.session_state:
    st.session_state.inference = False
        
# Load all data
data = load_all_modalities()
t2_data, t2 = data["t2"]
adc_data, adc = data["adc"]
hbv_data, hbv = data["hbv"]


# Use smallest depth across modalities for slider
max_slices = min(t2.shape[0], adc.shape[0], hbv.shape[0])


# Title
st.markdown("<h1 style='text-align: center;'>Prostate Cancer Segmentation Viewer</h1>", unsafe_allow_html=True)
st.write("")


# Sidebar navigation to switch between MRI and Results view
nav_options = ["MRI Viewer", "Results"]
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Select View", nav_options, index=0)


# MRI view
if selected_page == "MRI Viewer":
    
    # Slice to select slices across modalities
    slice_idx = st.sidebar.slider("Slice Index", min_value=0, max_value=max_slices-1, value=0, help="Select slice to display across modalities")
    
    # Column ratio for displaying images proportionally
    ratio_adc_to_t2 = adc.shape[1] / t2.shape[1]
    width_offset = 0.16
    col1, col2, col3 = st.columns([ratio_adc_to_t2 + width_offset, ratio_adc_to_t2, ratio_adc_to_t2], vertical_alignment='center')

    # Display images
    modalities = [("T2-weighted MRI", t2_data, t2),
                  ("ADC MRI", adc_data, adc),
                  ("HBV MRI", hbv_data, hbv)]
    
    for col, (label, data, img) in zip([col1, col2, col3], modalities):
        with col:
            st.markdown(f"<h4 style='text-align: center;'>{label}</h4>", unsafe_allow_html=True)
            st.image(img[slice_idx, :, :], use_column_width=True)

            # Display metadata in sidebar
            with st.sidebar.expander(label, expanded=False):
                for key, val in data.items():
                    st.write(f"**{key}:** {val}")

# Results View
elif selected_page == "Results":
    if st.sidebar.button("Run Inference"):
        st.session_state.inference = load_inference()
      
    if st.session_state.inference:
        # Load model confidence score
        with open(SCORE_PATH, "r") as f:
            score = json.load(f)
        confidence_score = float(score) if isinstance(score, (int, float)) else None
     
        
        # Display confidence score
        if confidence_score is not None:
            with st.sidebar.container(border=True):
                st.markdown(f"**Confidence Score**: {confidence_score:.2f}")
        else:
            st.sidebar.warning("Error Loading Confidence Score")
        
        # Load segmentation mask and metadata
        mask_data, mask = load_segmentation()

        # Display segmentation results 
        display_results(t2, mask, mask_data)
    
    else:
        st.warning("Please run inference to view results")
        
        
    

