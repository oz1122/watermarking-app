import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.decomposition import PCA
import pickle

st.set_page_config(page_title="Gabungan Watermarking SVD & PCA", layout="centered")
st.title("🔐 Aplikasi Watermarking Gambar Berwarna (SVD & PCA)")

# ===== Fungsi SVD =====

def embed_watermark_svd(image, watermark, alpha=0.1):
    h, w = image.shape[:2]
    watermark = cv2.resize(watermark, (w, h))
    watermarked = np.zeros_like(image)
    Uwm_list, Vtwm_list = [], []
    for c in range(3):
        img_ch = image[:, :, c].astype(float)
        wm_ch = watermark[:, :, c].astype(float)
        U_img, S_img, Vt_img = np.linalg.svd(img_ch, full_matrices=False)
        U_wm, S_wm, Vt_wm = np.linalg.svd(wm_ch, full_matrices=False)
        S_new = S_img + alpha * S_wm
        watermarked_ch = np.dot(U_img, np.dot(np.diag(S_new), Vt_img))
        watermarked[:, :, c] = np.clip(watermarked_ch, 0, 255).astype(np.uint8)
        Uwm_list.append(U_wm)
        Vtwm_list.append(Vt_wm)
    metadata = {'U_wm': Uwm_list, 'Vt_wm': Vtwm_list, 'alpha': alpha, 'original_shape': image.shape}
    return watermarked, metadata

def extract_watermark_svd(original, watermarked, metadata):
    extracted = np.zeros_like(original)
    alpha = metadata['alpha']
    Uwm_list = metadata['U_wm']
    Vtwm_list = metadata['Vt_wm']
    for c in range(3):
        orig_ch = original[:, :, c].astype(float)
        water_ch = watermarked[:, :, c].astype(float)
        _, S_orig, _ = np.linalg.svd(orig_ch, full_matrices=False)
        _, S_water, _ = np.linalg.svd(water_ch, full_matrices=False)
        S_wm = (S_water - S_orig) / alpha
        wm_ch = np.dot(Uwm_list[c], np.dot(np.diag(S_wm), Vtwm_list[c]))
        extracted[:, :, c] = np.clip(wm_ch, 0, 255).astype(np.uint8)
    return extracted

# ===== Fungsi PCA =====

def embed_watermark_pca(image, watermark, alpha=0.1):
    h, w = image.shape[:2]
    watermark = cv2.resize(watermark, (w, h))
    watermarked = np.zeros_like(image)
    for c in range(3):
        img_ch = image[:, :, c].astype(float).flatten().reshape(-1, 1)
        wm_ch = watermark[:, :, c].astype(float).flatten().reshape(-1, 1)
        pca_img = PCA(n_components=1)
        img_ch_pca = pca_img.fit_transform(img_ch)
        wm_ch_pca = PCA(n_components=1).fit_transform(wm_ch)
        img_ch_pca += alpha * wm_ch_pca
        watermarked_ch = pca_img.inverse_transform(img_ch_pca).reshape(h, w)
        watermarked[:, :, c] = np.clip(watermarked_ch, 0, 255).astype(np.uint8)
    metadata = {'alpha': alpha, 'original_shape': image.shape, 'method': 'PCA'}
    return watermarked, metadata

def extract_watermark_pca(original, watermarked, alpha=0.1):
    h, w = original.shape[:2]
    extracted = np.zeros_like(original)
    for c in range(3):
        orig_ch = original[:, :, c].astype(float).flatten().reshape(-1, 1)
        water_ch = watermarked[:, :, c].astype(float).flatten().reshape(-1, 1)
        pca = PCA(n_components=1)
        orig_ch_pca = pca.fit_transform(orig_ch)
        water_ch_pca = PCA(n_components=1).fit_transform(water_ch)
        wm_ch_pca = (water_ch_pca - orig_ch_pca) / alpha
        extracted_ch = pca.inverse_transform(wm_ch_pca).reshape(h, w)
        extracted[:, :, c] = np.clip(extracted_ch, 0, 255).astype(np.uint8)
    return extracted

# ===== Utilitas =====

def convert_to_bytes(img_array):
    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def save_metadata(metadata):
    return pickle.dumps(metadata)

def load_metadata(uploaded_file):
    try:
        data = pickle.load(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Gagal memuat metadata: {e}")
        return None

# ===== UI =====

method = st.sidebar.selectbox("Pilih Metode Watermarking", ["SVD", "PCA"])
option = st.sidebar.radio("Pilih Tindakan", ("Sisipkan Watermark", "Ekstrak Watermark"))

if option == "Sisipkan Watermark":
    st.header(f"1️⃣ Upload Gambar (Metode {method})")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("📤 Gambar Utama", type=["jpg", "png", "jpeg"], key="img")
        if uploaded_image:
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
            st.image(image, caption="🖼️ Gambar Utama", use_container_width=True, channels="BGR")
        else:
            image = None
    with col2:
        uploaded_watermark = st.file_uploader("📤 Gambar Watermark", type=["jpg", "png", "jpeg"], key="wm")
        if uploaded_watermark:
            watermark = cv2.imdecode(np.frombuffer(uploaded_watermark.read(), np.uint8), 1)
            st.image(watermark, caption="💧 Watermark", use_container_width=True, channels="BGR")
        else:
            watermark = None
    st.divider()
    if image is not None and watermark is not None:
        st.header("2️⃣ Aksi Watermarking")
        alpha = st.slider("Alpha (intensitas watermark)", 0.01, 1.0, 0.1, step=0.01)
        if st.button(f"📌 Sisipkan Watermark dengan {method}"):
            if method == "SVD":
                watermarked, metadata = embed_watermark_svd(image, watermark, alpha)
                st.image(watermarked, caption="✅ Gambar Setelah Disisipi", use_container_width=True, channels="BGR")
                st.download_button("⬇️ Download Gambar Watermarked (PNG)", data=convert_to_bytes(watermarked), file_name="watermarked.png", mime="image/png")
                st.download_button("⬇️ Download Metadata (PKL)", data=save_metadata(metadata), file_name="metadata.pkl", mime="application/octet-stream")
            else:
                watermarked, metadata = embed_watermark_pca(image, watermark, alpha)
                st.image(watermarked, caption="✅ Gambar Setelah Disisipi", use_container_width=True, channels="BGR")
                st.download_button("⬇️ Download Gambar Watermarked", data=convert_to_bytes(watermarked), file_name="watermarked_output.png", mime="image/png")

elif option == "Ekstrak Watermark":
    st.header(f"1️⃣ Upload Gambar Watermarked (Metode {method})")
    if method == "SVD":
        uploaded_watermarked_image = st.file_uploader("📤 Gambar Watermarked (PNG/JPG)", type=["jpg", "png", "jpeg"], key="watermarked_img")
        uploaded_metadata = st.file_uploader("📤 Upload Metadata (.pkl)", type=["pkl"], key="metadata_file")
        if uploaded_watermarked_image and uploaded_metadata:
            watermarked_image = cv2.imdecode(np.frombuffer(uploaded_watermarked_image.read(), np.uint8), 1)
            metadata = load_metadata(uploaded_metadata)
            st.image(watermarked_image, caption="🖼️ Gambar Watermarked", use_container_width=True, channels="BGR")
        else:
            watermarked_image = None
            metadata = None
    else:
        uploaded_watermarked_image = st.file_uploader("📤 Gambar Watermarked (format gambar)", type=["jpg", "png", "jpeg"], key="watermarked_img_pca")
        if uploaded_watermarked_image:
            watermarked_image = cv2.imdecode(np.frombuffer(uploaded_watermarked_image.read(), np.uint8), 1)
            metadata = None
            st.image(watermarked_image, caption="🖼️ Gambar Watermarked", use_container_width=True, channels="BGR")
        else:
            watermarked_image = None

    st.divider()
    if watermarked_image is not None:
        st.header("2️⃣ Upload Gambar Asli (untuk ekstraksi)")
        uploaded_original_image = st.file_uploader("📤 Gambar Asli", type=["jpg", "png", "jpeg"], key="original_img")
        if uploaded_original_image:
            original_image = cv2.imdecode(np.frombuffer(uploaded_original_image.read(), np.uint8), 1)
            if method == "SVD":
                if metadata is None:
                    st.error("Metadata tidak ditemukan! Harap upload file metadata (.pkl).")
                else:
                    extracted = extract_watermark_svd(original_image, watermarked_image, metadata)
                    st.image(extracted, caption="🎯 Watermark yang Diekstraksi", use_container_width=True, channels="BGR")
                    st.download_button("⬇️ Download Watermark", data=convert_to_bytes(extracted), file_name="extracted_watermark.png", mime="image/png")
