import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Klasifikasi Sampah - NPM 50422073",
    page_icon="♻️",
    layout="centered"
)

# --- STYLE CSS (Agar tampilan lebih rapi) ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DOWNLOAD MODEL ---
MODEL_PATH = "model_sampah.h5"

@st.cache_data
def download_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "18hc8WHannK0asctqfjgyFiDp2ZxbRsDY"
        url = f'https://drive.google.com/file/d/18hc8WHannK0asctqfjgyFiDp2ZxbRsDY/view?usp=drive_link'
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")

download_model()

# --- LOAD MODEL (Penanganan Error Keras 3) ---
@st.cache_resource
def load_ml_model():
    import tensorflow as tf
    try:
        # Menggunakan compile=False untuk menghindari error pada optimizer/quantization
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ **Error Loading Model:** {e}")
        st.info("Saran: Pastikan requirements.txt sudah berisi 'tensorflow==2.16.1'")
        return None

model = load_ml_model()

# --- FUNGSI PREDIKSI ---
def predict_image(image):
    # Preprocessing
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Proses Prediksi
    prediction = model.predict(img)
    confidence = float(prediction[0][0])
    
    # Logika Label (Sigmoid)
    if confidence > 0.5:
        label = "ANORGANIK"
        prob = confidence * 100
    else:
        label = "ORGANIK"
        prob = (1 - confidence) * 100
        
    return label, prob

# --- USER INTERFACE ---
st.title("♻️ Klasifikasi Sampah CNN")
st.write("Website Pendeteksi Sampah Organik & Anorganik")
st.info("Skripsi Adam Fawwaz Aydin 50422073")

if model is not None:
    # Menggunakan Menu Tab
    tab1, tab2 = st.tabs(["📤 Upload File", "📷 Ambil Foto"])

    # --- Bagian Upload ---
    with tab1:
        uploaded_file = st.file_uploader("Pilih Gambar Sampah...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar yang Diupload", use_container_width=True)
            
            with st.spinner("Sedang Menganalisis..."):
                label, prob = predict_image(img)
                
                st.divider()
                st.subheader(f"Hasil: {label}")
                st.write(f"Tingkat Keyakinan: **{prob:.2f}%**")
                
                # Warna progress bar berdasarkan label
                st.progress(int(prob))
                if label == "ORGANIK":
                    st.success("Sampah ini termasuk golongan yang dapat terurai.")
                else:
                    st.warning("Sampah ini termasuk golongan yang sulit terurai.")

    # --- Bagian Kamera ---
    with tab2:
        camera_file = st.camera_input("Arahkan kamera ke sampah")
        if camera_file:
            img_cam = Image.open(camera_file)
            
            with st.spinner("Menganalisis hasil foto..."):
                label, prob = predict_image(img_cam)
                
                st.divider()
                st.subheader(f"Hasil: {label}")
                st.write(f"Tingkat Keyakinan: **{prob:.2f}%**")
                st.progress(int(prob))

else:
    st.warning("Aplikasi sedang dalam mode perbaikan (Model Error).")

# --- FOOTER ---
st.markdown("---")
st.caption("NPM: 50422073 | 2026")