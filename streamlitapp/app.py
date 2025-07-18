import streamlit as st
import os
from PIL import Image
from profile_functions import read_json_profile, profiler_visualization

# Define paths and load icon
dirname = os.path.dirname(__file__)
stelar_icon = os.path.join(dirname, 'icons/stelar_icon.jpg')
im = Image.open(stelar_icon)

# Configure Streamlit page
st.set_page_config(
    page_title="Stelar Data Profiler - Visualization",
    page_icon=im,
    layout="wide"
)

# App title with icon
st.title("🌟 Stelar Data Profiler - Data Visualization")
st.markdown("Upload a previously generated JSON profile file to explore its visualization.")

# Upload form
with st.form("visualization_form"):
    uploaded_file = st.file_uploader("Upload your profiling `.json` file", type=["json"])
    submitted = st.form_submit_button("Visualize Profile")

# Handle submission
if submitted or uploaded_file is not None:
    if uploaded_file is not None:
        try:
            config_dict = read_json_profile(uploaded_file.read())
            profiler_visualization(config_dict)
        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {e}")
    else:
        if submitted:
            st.warning("⚠️ Please upload a JSON file to proceed.")