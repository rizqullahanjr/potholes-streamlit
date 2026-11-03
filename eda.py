import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data(show_spinner="Memindai dataset dan mengekstrak metadata citra...")
def load_image_metadata(dataset_path):
    '''
    Fungsi Data Loading & Metadata Extraction diambil langsung dari Code Cell Notebook Modelling.
    Mengekstrak filepath, label, width, height, aspect ratio, color mode, dan channel count.
    '''
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    records = []
    corrupted_files = []

    for label_name in ['normal', 'potholes']:
        folder_path = os.path.join(dataset_path, label_name)
        if os.path.exists(folder_path):
            file_list = sorted(glob.glob(os.path.join(folder_path, '*')))
            for fpath in file_list:
                if fpath.lower().endswith(valid_exts):
                    try:
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
                    except Exception as e:
                        corrupted_files.append((fpath, str(e)))

    df = pd.DataFrame(records)
    return df, corrupted_files


def find_default_dataset_path():
    '''
    Mendeteksi path folder dataset secara dinamis (lokal, subfolder, atau root).
    '''
    candidate_paths = [
        'dataset',
        '../dataset',
        'data',
        '../data',
        'pothole-detection-dataset',
        '/content/dataset'
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isdir(p):
            # Pastikan ada folder normal atau potholes di dalamnya
            subdirs = os.listdir(p)
            if 'normal' in subdirs or 'potholes' in subdirs:
                return p
    return 'dataset'


def run():
    '''
    Fungsi utama dashboard EDA yang mengeksekusi logika visualisasi
    langsung dari cell notebook permodelan utama.
    '''
    st.title('Exploratory Data Analysis - Pothole Detection')
    st.subheader('Analisis Karakteristik Visual & Metadata Citra Jalan')

    st.markdown('''
    Dashboard ini mengeksekusi langsung tahapan **Exploratory Data Analysis (EDA)** dari notebook permodelan utama:
    - **Distribusi Kelas Target (`normal` vs `potholes`)**
    - **Visualisasi Sampel Citra & Karakteristik Permukaan Jalan**
    - **Analisis Resolusi (Width, Height, & Aspect Ratio)**
    - **Analisis Color Mode & Sebaran Intensitas Piksel RGB**
    ''')

    # -------------------------------------------------------------------------
    # 1. DATA LOADING & PATH CONFIGURATION
    # -------------------------------------------------------------------------
    default_path = find_default_dataset_path()
    
    with st.expander('Pengaturan Lokasi Folder Dataset', expanded=False):
        custom_path = st.text_input(
            'Path Direktori Dataset (berisi folder `normal` dan `potholes`):',
            value=default_path
        )

    dataset_path = custom_path.strip()

    if not os.path.exists(dataset_path):
        st.warning(
            f"Direktori `{dataset_path}` tidak ditemukan di repositori ini. "
            "Jika dijalankan di lokal, letakkan folder `dataset/` (berisi folder `normal/` dan `potholes/`) di root proyek."
        )
        return

    # Load metadata langsung dari file citra
    df, corrupted = load_image_metadata(dataset_path)

    if df.empty:
        st.error(
            f"Tidak ditemukan citra valid di dalam folder `{dataset_path}/normal` atau `{dataset_path}/potholes`."
        )
        return

    st.success(
        f"Berhasil memindai **{len(df):,}** citra dari folder `{dataset_path}` "
        f"({df['label'].value_counts().to_dict()}) | Corrupted: {len(corrupted)}"
    )

    # Menampilkan 5 baris pertama data metadata mentah
    with st.expander('Lihat 5 Baris Pertama Metadata Citra (DataFrame `df`)', expanded=False):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown('---')

    # -------------------------------------------------------------------------
    # 2. DISTRIBUSI KELAS TARGET (Sesuai Cell EDA 1 Notebook)
    # -------------------------------------------------------------------------
    st.write('### 1. Distribusi Kelas Target (`normal` vs `potholes`)')

    class_counts = df['label'].value_counts()
    class_pct = (df['label'].value_counts(normalize=True) * 100).round(2)
    class_summary_df = pd.DataFrame({
        'Kelas Label': class_counts.index,
        'Jumlah Citra': class_counts.values,
        'Persentase (%)': class_pct.values
    })
    st.dataframe(class_summary_df, use_container_width=True)

    fig1, ax1 = plt.subplots(1, 2, figsize=(13, 4.5))

    # Plot 1: Countplot jumlah citra per kelas
    sns.countplot(
        data=df,
        x='label',
        order=['normal', 'potholes'],
        hue='label',
        legend=False,
        palette=['#2b5c8f', '#d95f02'],
        ax=ax1[0]
    )
    ax1[0].set_title('Distribusi Jumlah Citra per Kelas', fontsize=11.5, fontweight='bold')
    ax1[0].set_xlabel('Kelas Label')
    ax1[0].set_ylabel('Jumlah Citra')
    ax1[0].set_ylim(0, max(class_counts.values) * 1.18)

    for p in ax1[0].patches:
        h = int(p.get_height())
        pct = (h / len(df)) * 100
        ax1[0].annotate(
            f'{h} ({pct:.2f}%)',
            (p.get_x() + p.get_width() / 2., h),
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    # Plot 2: Pie chart proporsi kelas
    labels_pie = [f"{lbl}\n({class_counts.get(lbl, 0)} citra)" for lbl in class_counts.index]
    ax1[1].pie(
        class_counts.values,
        labels=labels_pie,
        autopct='%1.2f%%',
        startangle=90,
        colors=['#2b5c8f', '#d95f02'][:len(class_counts)],
        explode=[0.03] * len(class_counts),
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax1[1].set_title('Proporsi Persentase Kelas Dataset', fontsize=11.5, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown('''
    **Insight Distribusi Kelas:**
    - Perbandingan jumlah citra antara kelas **`normal`** dan **`potholes`** tergolong **seimbang (*balanced*)**.
    - Karena distribusinya seimbang, metrik akurasi valid digunakan berdampingan dengan fokus utama pada **Recall** dan **F1-Score** kelas `potholes` (untuk meminimalkan risiko kecelakaan akibat lubang yang tidak terdeteksi).
    ''')

    st.markdown('---')

    # -------------------------------------------------------------------------
    # 3. SAMPEL CITRA TIAP KELAS
    # -------------------------------------------------------------------------
    st.write('### 2. Visualisasi Sampel Citra Asli & Karakteristik Visual Tiap Kelas')

    samples_per_class = st.slider('Jumlah Sampel Citra yang Ditampilkan per Kelas:', min_value=2, max_value=6, value=4)

    for label in ['normal', 'potholes']:
        subset = df[df['label'] == label]
        if len(subset) == 0:
            continue
        
        sample_subset = subset.sample(n=min(len(subset), samples_per_class), random_state=42).reset_index(drop=True)
        st.write(f"**Sampel Kelas: `{label.upper()}`**")

        cols = st.columns(len(sample_subset))
        for idx, row in sample_subset.iterrows():
            with cols[idx]:
                try:
                    with Image.open(row['images']) as img:
                        st.image(
                            img,
                            caption=f"{row['filename']}\n({row['width']}x{row['height']} px | {row['color_mode']})",
                            use_container_width=True
                        )
                except Exception as err:
                    st.caption(f"Gagal memuat {row['filename']}: {err}")

    st.markdown('''
    **Karakteristik Visual yang Diamati:**
    1. **Kelas `normal`**: Permukaan aspal rata dengan pola tekstur homogen dan garis marka jalan kontinu.
    2. **Kelas `potholes`**: Memiliki tepi patahan aspal tajam tidak beraturan (*irregular edges*), bayangan cekungan gelap, atau genangan air hujan yang memantulkan cahaya.
    ''')

    st.markdown('---')

    # -------------------------------------------------------------------------
    # 4. ANALISIS DIMENSI & ASPECT RATIO
    # -------------------------------------------------------------------------
    st.write('### 3. Analisis Dimensi Resolusi (Width, Height, & Aspect Ratio)')

    dim_summary = df.groupby('label').agg(
        Jumlah_Citra=('images', 'count'),
        Min_Width=('width', 'min'),
        Mean_Width=('width', lambda x: round(x.mean(), 1)),
        Median_Width=('width', 'median'),
        Max_Width=('width', 'max'),
        Min_Height=('height', 'min'),
        Mean_Height=('height', lambda x: round(x.mean(), 1)),
        Median_Height=('height', 'median'),
        Max_Height=('height', 'max'),
        Mean_Aspect_Ratio=('aspect_ratio', lambda x: round(x.mean(), 2))
    ).reset_index()

    st.dataframe(dim_summary, use_container_width=True)

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot Width vs Height
    sns.scatterplot(
        data=df,
        x='width',
        y='height',
        hue='label',
        style='label',
        palette=['#2b5c8f', '#d95f02'],
        alpha=0.75,
        s=60,
        ax=ax2[0]
    )
    ax2[0].axvline(224, color='crimson', linestyle='--', linewidth=1.5, label='Target Resize (224x224 px)')
    ax2[0].axhline(224, color='crimson', linestyle='--', linewidth=1.5)
    ax2[0].set_title('Sebaran Resolusi Asli Citra (Width vs Height)', fontsize=11.5, fontweight='bold')
    ax2[0].set_xlabel('Width / Lebar (piksel)')
    ax2[0].set_ylabel('Height / Tinggi (piksel)')
    ax2[0].legend(title='Kelas / Target')

    # Histogram Aspect Ratio
    sns.histplot(
        data=df,
        x='aspect_ratio',
        hue='label',
        kde=True,
        bins=25,
        palette=['#2b5c8f', '#d95f02'],
        ax=ax2[1]
    )
    ax2[1].axvline(1.0, color='crimson', linestyle='--', linewidth=1.5, label='Rasio 1:1 (Square 224x224)')
    ax2[1].set_title('Distribusi Aspect Ratio Citra (Width / Height)', fontsize=11.5, fontweight='bold')
    ax2[1].set_xlabel('Aspect Ratio (Width / Height)')
    ax2[1].set_ylabel('Jumlah Citra')
    ax2[1].legend()

    plt.tight_layout()
    st.pyplot(fig2)

    st.info(
        "**Alasan Wajib Dilakukan Image Resizing (224x224 px):** "
        "Ukuran resolusi citra sangat bervariasi (mulai dari `<200 px` hingga `>3000 px`), "
        "sementara arsitektur CNN dan model MobileNetV2 mewajibkan dimensi tensor input yang identik "
        "`(batch_size, 224, 224, 3)`. Selain itu, penyeragaman ke 224x224 px menjaga detail tekstur lubang "
        "tetap jelas sekaligus menjaga efisiensi komputasi dan memori GPU."
    )

    st.markdown('---')

    # -------------------------------------------------------------------------
    # 5. ANALISIS COLOR CHANNEL & PIXEL INTENSITY
    # -------------------------------------------------------------------------
    st.write('### 4. Analisis Color Channel & Distribusi Intensitas Piksel RGB')

    channel_dist = df.groupby(['label', 'color_mode', 'channels']).size().reset_index(name='jumlah_citra')
    st.dataframe(channel_dist, use_container_width=True)

    # Mengambil sampel representatif untuk kalkulasi intensitas warna RGB
    sample_size = min(len(df), 120)
    sample_rgb_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
    )

    rgb_records = []
    for _, row in sample_rgb_df.iterrows():
        try:
            with Image.open(row['images']) as img:
                img_rgb = img.convert('RGB').resize((224, 224))
                arr = np.array(img_rgb, dtype=np.float32)
                rgb_records.append({
                    'label': row['label'],
                    'Red_Mean': arr[:, :, 0].mean(),
                    'Green_Mean': arr[:, :, 1].mean(),
                    'Blue_Mean': arr[:, :, 2].mean(),
                    'Brightness_Mean': arr.mean()
                })
        except Exception:
            pass

    if rgb_records:
        df_rgb = pd.DataFrame(rgb_records)
        rgb_summary = df_rgb.groupby('label')[['Red_Mean', 'Green_Mean', 'Blue_Mean', 'Brightness_Mean']].agg(['mean', 'median']).round(2)
        st.dataframe(rgb_summary, use_container_width=True)

        fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.2))
        channel_configs = [
            ('Red_Mean', 'Distribusi Intensitas Channel RED', 'crimson'),
            ('Green_Mean', 'Distribusi Intensitas Channel GREEN', 'seagreen'),
            ('Blue_Mean', 'Distribusi Intensitas Channel BLUE', 'royalblue')
        ]

        for ax, (col, title, _) in zip(axes3, channel_configs):
            sns.kdeplot(
                data=df_rgb,
                x=col,
                hue='label',
                fill=True,
                common_norm=False,
                palette=['#2b5c8f', '#d95f02'],
                alpha=0.35,
                ax=ax
            )
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Rata-rata Intensitas Piksel (0 - 255)')
            ax.set_ylabel('Density')

        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown('''
    **Kesimpulan Analisis Color Channel:**
    - Sebagian besar citra menggunakan format 3-channel (RGB), dengan sedikit variasi citra grayscale (1 channel) atau RGBA (4 channel). Oleh sebab itu, fungsi preprocessing wajib mengonversi citra secara eksplisit ke RGB (`.convert('RGB')` / `color_mode='rgb'`).
    - Kurva intensitas warna global antara `normal` dan `potholes` saling beririsan karena kedua kelas sama-sama didominasi aspal dan tanah. Hal ini menegaskan mengapa jaringan konvolusi (CNN) dibutuhkan untuk mengekstrak fitur spasial lokal, bukan sekadar melihat intensitas warna rata-rata.
    ''')


if __name__ == '__main__':
    run()