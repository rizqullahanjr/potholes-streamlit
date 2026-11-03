import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Model & Mapping Kelas
model_path = '../best_pothole_model.keras'
best_pothole_model = load_model(model_path, compile=False)
class_names = {0: 'normal', 1: 'potholes'}


def run():
    '''
    Fungsi untuk menjalankan form dan proses prediksi citra kondisi jalan raya.
    '''
    st.title('Pothole Road Damage Detection')
    st.subheader('Formulir Upload Citra Jalan & Prediksi Klasifikasi CNN')

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
            st.warning('Silakan unggah file gambar permukaan jalan terlebih dahulu.')
            return

        # Membaca citra asli dan konversi ke 3-channel RGB
        raw_image = Image.open(uploaded_file).convert('RGB')
        orig_w, orig_h = raw_image.size

        col1, col2 = st.columns(2)
        with col1:
            st.write('### Citra Jalan yang Diunggah:')
            st.image(raw_image, caption=f'Lokasi: {lokasi_jalan} ({orig_w}x{orig_h} px)', use_container_width=True)

        # Preprocessing citra: Resizing ke (224, 224) dan Scaling (1./255.)
        img_resized = raw_image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Prediksi menggunakan model CNN terbaik
        prob_pothole = float(best_pothole_model.predict(img_batch, verbose=0)[0][0])
        pred_binary = 1 if prob_pothole >= 0.5 else 0
        pred_label_str = class_names[pred_binary]
        confidence_pct = (prob_pothole if pred_binary == 1 else (1.0 - prob_pothole)) * 100.0

        with col2:
            st.write('### Ringkasan Hasil Prediksi:')
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
                f'(Confidence: **{confidence_pct:.2f}%**)'
            )
            st.write('**Rekomendasi Tindakan Operasional:**')
            st.write('- **Tindakan Darurat:** Segera pasang rambu peringatan di lokasi kerusakan jalan untuk mencegah kecelakaan.')
            st.write('- **Tindak Lanjut Perbaikan:** Masukkan ruas jalan ini ke dalam antrean prioritas tim pemeliharaan untuk segera ditambal.')
        else:
            st.success(
                f'**Hasil Prediksi: `{pred_label_str}` — Kondisi Permukaan Jalan Mulus / Normal** '
                f'(Confidence: **{confidence_pct:.2f}%**)'
            )
            st.write('**Rekomendasi Tindakan Operasional:**')
            st.write('- **Penyaringan Laporan:** Permukaan jalan terpantau normal (`normal`). Tidak memerlukan tindakan perbaikan darurat.')
            st.write('- **Pemeliharaan Rutin:** Simpan dokumentasi ke dalam database pemantauan berkala.')


if __name__ == '__main__':
    run()