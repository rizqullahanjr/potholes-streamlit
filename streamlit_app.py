import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title='Pothole Road Damage Detection - CNN',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.sidebar.title('Navigasi Aplikasi')
page = st.sidebar.selectbox('Pilih Halaman:', ['EDA', 'Prediction'])

if page == 'EDA':
    eda.run()
else:
    prediction.run()
