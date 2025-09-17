# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
from detector import DarkCircleDetector

st.set_page_config(page_title="Dark Circle Detector")
st.title("Dark Circle Detector")

@st.cache_resource 
def load_detector():
    return DarkCircleDetector()

detector = load_detector()

# Image upload and processing
uploaded = st.file_uploader("Upload frontal face image(s)", type=['jpg','jpeg','png','heif'])
if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    img_bgr = np.array(pil)[:, :, ::-1]                 # Convert RGB to BGR for OpenCV
    st.subheader("Original")
    st.image(pil, use_container_width=True)             # Display original image

    with st.spinner("Processing..."):
        res = detector.process_image(img_bgr, edge_method=None)

    if not res.get('success'):
        st.error("Failed: " + str(res.get('error')))
    else:
        ann_rgb = res['annotated_image'][:, :, ::-1]

        # Display annotated image
        st.subheader("Annotated result")
        st.image(ann_rgb, use_container_width=True)

        # Summary of results
        st.subheader("Faces detected")
        for i, f in enumerate(res['faces']):
            cols = st.columns(4)

            cols[0].markdown(f"**Face #{i+1}**")                                # Face ID
            conf = f.get('confidence')                                          # Confidence Score
        
            cols[1].write(f"Confidence: {conf:.3f}" if conf else "N/A")
            lsev = f['left_severity']; rsev = f['right_severity']               # Severity Scores

            cols[2].write(f"Left: {lsev[1]} ({lsev[0]:.2f})")                   # Left eye severity

            cols[3].write(f"Right: {rsev[1]} ({rsev[0]:.2f})")                  # Right eye severity

        # Download annotated image
        buf = io.BytesIO()
        Image.fromarray(ann_rgb).save(buf, format='PNG')
        st.download_button(
            "Download annotated image",
            buf.getvalue(),
            file_name="annotated_mtcnn.png",
            mime="image/png"
        )

# Sidebar - info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app detects and analyzes dark circles under and around the eyes using advanced computer vision techniques.\n\n"
        "Features:\n"
        "- Face and eye detection with MTCNN Model\n"
        "- Dark circle/dark spot analysis\n"
        "- Severity scoring and clear visualization\n\n"
        "Created with [OpenCV](https://opencv.org/) and [Streamlit](https://streamlit.io/).\n\n"
        "_Demo build by MANTHAN JADAV_"
    )

