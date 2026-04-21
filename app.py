from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from src.segmentation_project import VehicleSegmentationPipeline


st.set_page_config(page_title="Vehicle Segmentation UI", layout="wide")

st.title("Vehicle Segmentation and Counting")
st.caption(
    "Upload or paste a screenshot (for example from Google Maps), then run the segmentation pipeline."
)


@st.cache_resource
def get_pipeline(model_path: str, confidence: float) -> VehicleSegmentationPipeline:
    return VehicleSegmentationPipeline(model_path=model_path, confidence=confidence)


with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="yolov8n-seg.pt")
    confidence = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    st.markdown("Vehicle classes counted: car, bus, truck (COCO IDs 2, 5, 7)")


uploaded_file = st.file_uploader(
    "Paste or upload screenshot",
    type=["png", "jpg", "jpeg", "webp"],
    help="In most browsers you can click here and paste directly with Cmd+V.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image_rgb, use_container_width=True)

    if st.button("Run segmentation", type="primary"):
        pipeline = get_pipeline(model_path=model_path, confidence=confidence)
        output = pipeline.segment_and_count(image_rgb)

        overlay_rgb = output["overlay_rgb"]
        binary_mask = output["binary_mask"]
        vehicle_count = output["vehicle_count"]

        with col2:
            st.subheader("Segmented Output")
            st.image(overlay_rgb, use_container_width=True)

        st.metric("Detected vehicles", int(vehicle_count))
        st.subheader("Binary Vehicle Mask")
        st.image(binary_mask, clamp=True, use_container_width=True)

        mask_image = Image.fromarray(binary_mask)
        buf = BytesIO()
        mask_image.save(buf, format="PNG")
        st.download_button(
            label="Download mask (PNG)",
            data=buf.getvalue(),
            file_name="vehicle_mask.png",
            mime="image/png",
        )
else:
    st.info("Upload or paste an image to start.")
