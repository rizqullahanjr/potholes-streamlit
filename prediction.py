import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

@st.cache_resource
def load_pothole_model():
    """
    Memuat model CNN terbaik dengan path absolut langsung (saklek)
    mengikuti posisi file prediction.py berada, ditambah compile=False
    agar tidak error pada optimizer/state training saat inferensi.
    """

    model = load_model('best_pothole_model.keras', compile=False)
    return model

def run():
    '''
    Fungsi untuk menjalankan halaman prediksi citra kondisi jalan raya.
    Mengembalikan hasil prediksi ke dalam bentuk label string aslinya ('normal' atau 'potholes').
    '''
    st.title('Pothole Road Damage Detection')
    st.subheader('Formulir Unggah Citra Jalan & Prediksi Klasifikasi CNN')

    model = load_pothole_model()

    with st.form(key='pothole_prediction_form'):
        pelapor = st.text_input('Nama Petugas / Pelapor', value='Tim Inspeksi Lapangan')
        instansi = st.selectbox(
            'Kategori Pengguna / Instansi',
            (
                'Dinas Bina Marga / Dinas PUPR',
                'Pengelola Jalan Tol / Patroli Jalan Raya',
                'Developer Kawasan Perumahan (Kota Wisata / Alam Sutera / dll.)',
                'Laporan Masyarakat Umum'
            )
        )
        lokasi_jalan = st.text_input('Lokasi / Ruas Jalan', value='Jl. Raya Utama KM 12')
        uploaded_file = st.file_uploader(
            'Unggah Foto Permukaan Jalan (JPG / JPEG / PNG)',
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
        )
        submitted = st.form_submit_button('Predict')

    if submitted:
        if uploaded_file is None:
            st.warning('Silakan unggah file gambar permukaan jalan terlebih dahulu sebelum menekan tombol Predict.')
            return

        # Membaca citra asli yang diunggah user
        raw_image = Image.open(uploaded_file).convert('RGB')
        orig_w, orig_h = raw_image.size

        col1, col2 = st.columns(2)
        with col1:
            st.write('### Citra Jalan yang Diunggah:')
            st.image(raw_image, caption=f' Lokasi: {lokasi_jalan} (Resolusi Asli: {orig_w}x{orig_h} px)', use_container_width=True)

        # Preprocessing citra: Resizing ke (224, 224) dan Scaling (1./255.)
        img_resized = raw_image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        if model is None:
            st.error(
                'File model `best_pothole_model.keras` belum ditemukan di direktori aplikasi. '
                'Pastikan file model hasil training sudah diletakkan di folder deployment.'
            )
            return

        # Prediksi menggunakan model CNN terbaik
        prob_pothole = float(model.predict(img_batch, verbose=0)[0][0])
        pred_binary = 1 if prob_pothole >= 0.5 else 0

        # Mengubah kembali hasil prediksi numerik (0 / 1) menjadi label string asli ('normal' / 'potholes')
        pred_label_str = 'potholes' if pred_binary == 1 else 'normal'
        confidence_pct = (prob_pothole if pred_binary == 1 else (1.0 - prob_pothole)) * 100.0

        with col2:
            st.write('### Ringkasan Metadata & Hasil Prediksi:')
            info_df = pd.DataFrame([{
                'Pelapor': pelapor,
                'Instansi': instansi,
                'Lokasi Ruas Jalan': lokasi_jalan,
                'Resolusi Asli': f'{orig_w}x{orig_h} px',
                'Prediksi Kelas (Biner)': pred_binary,
                'Label Prediksi (String)': pred_label_str,
                'Probabilitas Potholes (%)': round(prob_pothole * 100, 2),
                'Confidence (%)': round(confidence_pct, 2)
            }])
            st.dataframe(info_df.T.rename(columns={0: 'Detail'}), use_container_width=True)

        st.markdown('---')
        if pred_label_str == 'potholes':
            st.error(
                f'**Hasil Prediksi: `{pred_label_str}` — Terdeteksi Jalan Berlubang!** '
                f'(Confidence: **{confidence_pct:.2f}%** | Probabilitas Potholes: **{prob_pothole*100:.2f}%**)'
            )
            st.write('**Rekomendasi Tindakan Operasional:**')
            st.write('- **Tindakan Darurat:** Segera pasang rambu atau tanda peringatan di titik lokasi kerusakan jalan untuk mencegah kecelakaan lalu lintas.')
            st.write('- **Tindak Lanjut Perbaikan:** Masukkan ruas jalan ini ke dalam antrean prioritas tim pemeliharaan (Bina Marga / PUPR / Pengelola Tol / Town Management) untuk segera ditambal.')
        else:
            st.success(
                f'**Hasil Prediksi: `{pred_label_str}` — Kondisi Permukaan Jalan Mulus / Normal** '
                f'(Confidence: **{confidence_pct:.2f}%** | Probabilitas Potholes: **{prob_pothole*100:.2f}%**)'
            )
            st.write('**Rekomendasi Tindakan Operasional:**')
            st.write('- **Penyaringan Laporan:** Permukaan jalan terpantau dalam kondisi baik (`normal`). Laporan ini tidak memerlukan pengiriman tim penambalan darurat.')
            st.write('- **Pemeliharaan Rutin:** Simpan dokumentasi ke dalam database pemantauan berkala.')

if __name__ == '__main__':
    run()
