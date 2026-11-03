import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_dataset(dataset_path='../dataset'):
    '''
    Memuat filepath gambar dan mengekstrak metadata dimensi serta color channel.
    '''
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    records = []

    for label_name in ['normal', 'potholes']:
        folder_path = os.path.join(dataset_path, label_name)
        if os.path.exists(folder_path):
            for fpath in sorted(glob.glob(os.path.join(folder_path, '*'))):
                if fpath.lower().endswith(valid_exts):
                    with Image.open(fpath) as img:
                        width, height = img.size
                        mode = img.mode
                        channels = len(img.getbands())
                    records.append({
                        'images': fpath,
                        'filename': os.path.basename(fpath),
                        'label': label_name,
                        'width': width,
                        'height': height,
                        'aspect_ratio': round(width / height, 4),
                        'color_mode': mode,
                        'channels': channels
                    })

    return pd.DataFrame(records)


def run():
    '''
    Fungsi untuk menjalankan dashboard Exploratory Data Analysis (EDA).
    '''
    st.title('Exploratory Data Analysis - Pothole Detection')
    st.subheader('Analisis Karakteristik Visual & Metadata Citra Jalan')

    df = load_dataset()

    st.write('### Sampel Metadata Citra')
    st.dataframe(df.head(10), use_container_width=True)
    st.dataframe(df.tail(10), use_container_width=True)

    # 1. Distribusi Kelas Target
    st.write('### 1. Distribusi Kelas Target (`normal` vs `potholes`)')
    class_counts = df['label'].value_counts()

    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(
        data=df, x='label', order=['normal', 'potholes'],
        hue='label', legend=False, palette=['#2b5c8f', '#d95f02'], ax=ax1[0]
    )
    ax1[0].set_title('Distribusi Jumlah Citra per Kelas')
    ax1[0].set_xlabel('Kelas Label')
    ax1[0].set_ylabel('Jumlah Citra')

    ax1[1].pie(
        class_counts.values,
        labels=class_counts.index,
        autopct='%1.2f%%',
        startangle=90,
        colors=['#2b5c8f', '#d95f02']
    )
    ax1[1].set_title('Proporsi Persentase Kelas Dataset')
    plt.tight_layout()
    st.pyplot(fig1)
    st.write(
        'Proporsi antara kelas `normal` dan `potholes` tergolong seimbang (*balanced*), '
    )

    # 2. Visualisasi Sampel Citra Tiap Kelas
    st.write('### 2. Visualisasi Sampel Citra Tiap Kelas')
    for label in ['normal', 'potholes']:
        st.write(f'**Kelas: `{label}`**')
        sample_df = df[df['label'] == label].sample(n=4, random_state=42).reset_index(drop=True)
        cols = st.columns(4)
        for idx, row in sample_df.iterrows():
            with cols[idx]:
                st.image(
                    Image.open(row['images']),
                    caption=f"{row['filename']} ({row['width']}x{row['height']} px)",
                    use_container_width=True
                )
    st.write(
            '1. `normal`:'
            '\n- Tekstur jalan yang mulus'
            '\n- Beberapa jalan besar (jalan nasional) yang ada garis marka jalan masih terlihat dengan baik'
            ''
            
            )
    st.write(
            '2. `potholes`:'
            '\n- Terdapat cekungan dan genangan air'
            '\n- Ada kontras pada pinggiran lubang akibat lapisan aspal yang rusak'
            '\n- Ketika ada cekungan jalan bolong terlihat bayangan gelap'
            '\n- Jika jalan bolong terdapat air pasti memantulkan cahaya '
    )

    # 3. Analisis Dimensi Resolusi & Aspect Ratio
    st.write('### 3. Analisis Dimensi Resolusi (Width, Height, & Aspect Ratio)')
    st.dataframe(df[['width', 'height', 'aspect_ratio', 'channels']].describe().T, use_container_width=True)

    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.scatterplot(
        data=df, x='width', y='height', hue='label',
        palette=['#2b5c8f', '#d95f02'], alpha=0.75, ax=ax2[0]
    )
    ax2[0].set_title('Sebaran Resolusi Asli Citra (Width vs Height)')
    ax2[0].legend()

    sns.histplot(
        data=df, x='aspect_ratio', hue='label', kde=True,
        bins=25, palette=['#2b5c8f', '#d95f02'], ax=ax2[1]
    )
    ax2[1].set_title('Distribusi Aspect Ratio Citra (Width / Height)')
    plt.tight_layout()
    st.pyplot(fig2)
    st.write(
        'Ukuran resolusi asli citra sangat beragam sehingga seluruh citra wajib di-*resize* '
        'ke ukuran **224x224 piksel** agar sesuai dengan input tensor CNN dan `MobileNetV2`.'
    )

    # 4. Distribusi Color Mode & Channel
    st.write('### 4. Distribusi Mode Warna & Color Channel')
    channel_dist = df.groupby(['label', 'color_mode', 'channels']).size().reset_index(name='jumlah_citra')
    st.dataframe(channel_dist, use_container_width=True)
    st.write(
        'Terdapat variasi jumlah kanal warna (1 hingga 4 channel), sehingga pada tahap *preprocessing* '
        'setiap citra dikonversi secara eksplisit ke **3-channel RGB**.'
    )


if __name__ == '__main__':
    run()