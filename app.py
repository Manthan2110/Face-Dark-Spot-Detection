# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
from detector import DarkCircleDetector
from streamlit_option_menu import option_menu  #added

st.set_page_config(page_title="Face Dark Spot & Eye Detection")
st.title("🧑‍⚕️ Face Dark Spot & Eye Detection")
st.write("An intelligent OpenCV-based app to detect faces, eyes, and dark spots.")

@st.cache_resource 
def load_detector():
    return DarkCircleDetector()

detector = load_detector()

#added: Selection for input method using streamlit-option-menu
input_method = option_menu(
    menu_title="Choose Input Method",  #added
    options=["Upload Image", "Use Webcam"],  #added
    icons=["folder-fill", "camera-fill"],  #added
    orientation="horizontal",  #added
    default_index=0  #added
)

img_bgr = None  #added: placeholder for the selected image

# Image upload option
if input_method == "Upload Image":  #added
    uploaded = st.file_uploader("Upload frontal face image(s)", type=['jpg','jpeg','png','heif'])
    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        img_bgr = np.array(pil)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

# Webcam capture option
elif input_method == "Use Webcam":  #added
    camera_img = st.camera_input("Capture your face")  #added
    if camera_img is not None:  #added
        pil = Image.open(camera_img).convert("RGB")  #added
        img_bgr = np.array(pil)[:, :, ::-1]  # Convert RGB to BGR for OpenCV #added

# Process if image is available
if img_bgr is not None:  #added
    with st.spinner("Processing..."):
        res = detector.process_image(img_bgr, edge_method=None)

    if not res.get('success'):
        st.error("Failed: " + str(res.get('error')))
    else:
        ann_rgb = res['annotated_image'][:, :, ::-1]

        # Side-by-side display: Original | Annotated
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil, caption="Original")
        with col2:
            st.image(ann_rgb, caption="Annotated")

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

            # # AI-based recommendations shown in expander below each face
            # # #added: severity-based tips
            # recommendations = {
            #         "Minimal": "Keep up good sleep and hydration. Use a lightweight moisturizer and sunscreen daily.",
            #         "Mild": "Try cold compresses and an eye cream with caffeine or hyaluronic acid. Improve sleep schedule (7–8 hrs).",
            #         "Moderate": "Use topical antioxidants (vitamin C), cool compresses, and consider dermatologist-recommended eye creams.",
            #         "Severe": "Consider consulting a dermatologist for targeted treatments. Maintain hydration, reduce alcohol, and use sun protection.",
            #         "Very Severe": "Seek professional dermatological advice — clinical treatments (peels, lasers) may be necessary along with lifestyle changes."
            #     }  # #added

            # # pick the worse severity for advice
            # worse = lsev if lsev[0] >= rsev[0] else rsev
            # with st.expander(f"Recommendations for Face #{i+1} - {worse[1]}"):  # #added
            #     st.write(recommendations.get(worse[1], "Maintain healthy habits: sleep, hydrate, sunscreen."))  # #added

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
        "- Choose between image upload or webcam capture\n"
        "- Face and eye detection with MTCNN Model\n"
        "- Dark circle/dark spot analysis\n"
        "- Severity scoring and clear visualization\n"
        "- Red Rectangles indicate dark spots\n\n"

        "Created with [OpenCV](https://opencv.org/) and [Streamlit](https://streamlit.io/).\n\n"
        "_Demo build by MANTHAN JADAV_"
    )
