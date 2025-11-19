"""Configuration file for input/output paths."""
import os

# Get path of current directory (works on all OS: Windows, Linux, Streamlit Cloud)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folder inside repo
INPUT_DIR = os.path.join(PROJECT_DIR, "test")

# Output folder inside repo
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output_debug")

# Segmentation mask path (relative)
SEG_PATH = os.path.join(
    OUTPUT_DIR, 
    "images", 
    "cspca-detection-map", 
    "cspca_detection_map.mha"
)

# Probability JSON (relative)
SCORE_PATH = os.path.join(
    OUTPUT_DIR, 
    "cspca-case-level-likelihood.json"
)
