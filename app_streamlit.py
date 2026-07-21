import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================

st.set_page_config(
    page_title="Klasifikasi Sampah CNN",
    page_icon="♻️",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stButton>button{
    width:100%;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# DOWNLOAD MODEL DARI HUGGING FACE
# ======================================================

@st.cache_resource
def get_model_path():
    return hf_hub_download(
        repo_id="mada19/ModelCNN",
        filename="model_sampah.h5"
    )

MODEL_PATH = get_model_path()

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_ml_model():

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        st.error(f"❌ Error Loading Model:\n\n{e}")
        st.stop()

model = load_ml_model()

# ======================================================
# PREDIKSI
# ======================================================

def predict_image(image):

    img = image.convert("RGB")

    img = np.array(img)

    img = cv2.resize(img, (150, 150))

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        label = "ANORGANIK"
        score = confidence * 100
    else:
        label = "ORGANIK"
        score = (1 - confidence) * 100

    return label, score

# ======================================================
# USER INTERFACE
# ======================================================

st.title("♻️ Klasifikasi Sampah Menggunakan CNN")

st.write("Website Pendeteksi Sampah Organik dan Anorganik")

st.info("Adam Fawwaz Aydin | NPM 50422073")

tab1, tab2 = st.tabs([
    "📂 Upload Gambar",
    "📷 Kamera"
])

# ======================================================
# TAB UPLOAD
# ======================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Gambar",
            use_container_width=True
        )

        if st.button("Prediksi"):

            with st.spinner("Sedang memproses..."):

                label, score = predict_image(image)

            st.success(f"Hasil : {label}")

            st.progress(min(int(score), 100))

            st.write(f"Confidence : **{score:.2f}%**")

            if label == "ORGANIK":
                st.success("Sampah termasuk Organik.")
            else:
                st.warning("Sampah termasuk Anorganik.")

# ======================================================
# TAB KAMERA
# ======================================================

with tab2:

    camera = st.camera_input("Ambil gambar")

    if camera is not None:

        image = Image.open(camera)

        with st.spinner("Sedang memproses..."):

            label, score = predict_image(image)

        st.success(f"Hasil : {label}")

        st.progress(min(int(score), 100))

        st.write(f"Confidence : **{score:.2f}%**")

        if label == "ORGANIK":
            st.success("Sampah termasuk Organik.")
        else:
            st.warning("Sampah termasuk Anorganik.")

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")
st.caption("Skripsi - Adam Fawwaz Aydin | 2026")
