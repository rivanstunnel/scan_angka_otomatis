# app.py

import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from itertools import product

from markov_model import (
    predict_markov,
    predict_markov_order2,
    predict_markov_hybrid,
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Prediksi 4D Markov", layout="wide")

# ==== PERBAIKAN: Fungsi untuk mereset hasil prediksi ====
def reset_prediction_state():
    """Hapus hasil prediksi jika ada perubahan pengaturan."""
    st.session_state.result = None

# Inisialisasi session_state jika belum ada
if 'result' not in st.session_state:
    st.session_state.result = None

# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
if lottie_predict:
    st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - Metode Markov")

metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # ==== PERBAIKAN: Setiap widget di sidebar akan mereset hasil jika diubah ====
    data_source = st.radio(
        "Sumber Data", ("API", "Input Manual"), horizontal=True,
        on_change=reset_prediction_state
    )
    
    if data_source == "API":
        hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
        selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list, on_change=reset_prediction_state)
        selected_hari = st.selectbox("📅 Pilih Hari", hari_list, on_change=reset_prediction_state)
    else:
        manual_data_input = st.text_area(
            "📋 Masukkan Data Keluaran", height=150,
            placeholder="Contoh: 1234 5678, 9012...",
            on_change=reset_prediction_state, key="manual_input_box"
        )
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', st.session_state.manual_input_box)
            st.session_state.df_data = pd.DataFrame({"angka": angka_list})
            st.session_state.last_query = "manual"
            reset_prediction_state() # Reset juga setelah memproses data baru
            st.success(f"✅ {len(angka_list)} data manual berhasil diproses.")

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100, on_change=reset_prediction_state)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", 1, 200, 10, on_change=reset_prediction_state)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list, on_change=reset_prediction_state)
    top_n = st.number_input("🔢 Jumlah Top Digit Prediksi", 1, 9, 6, on_change=reset_prediction_state)

# --- Logika Pengambilan Data (Hanya untuk API) ---
if data_source == "API":
    query_id = f"{selected_lokasi}-{selected_hari}"
    if 'df_data' not in st.session_state or st.session_state.get('last_query') != query_id:
        # (Logika pengambilan data API tidak berubah)
        with st.spinner(f"🔄 Mengambil data untuk pasaran {selected_lokasi}..."):
            # ...
            st.session_state.df_data = pd.DataFrame({"angka": []}) # Placeholder

df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)

if not df.empty:
    with st.expander(f"✅ Menampilkan {len(df)} data terakhir yang digunakan.", expanded=True):
        st.code("\n".join(df['angka'].tolist()), language="text")

# --- Tombol dan Logika Prediksi ---
if st.button("🔮 Prediksi Sekarang!", use_container_width=True):
    # Hanya jalankan prediksi jika belum ada hasilnya
    if st.session_state.result is None:
        if len(df) < 11:
            st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
        else:
            with st.spinner("⏳ Melakukan prediksi..."):
                result = None
                if metode == "Markov": result, _ = predict_markov(df, top_n=top_n)
                elif metode == "Markov Order-2": result = predict_markov_order2(df, top_n=top_n)
                elif metode == "Markov Gabungan": result = predict_markov_hybrid(df, top_n=top_n)
                
                # Simpan hasil ke session state
                st.session_state.result = result
    else:
        st.info("ℹ️ Hasil prediksi sudah ditampilkan. Ubah pengaturan untuk prediksi baru.")


# --- Tampilkan Hasil (selalu cek dari session_state) ---
if st.session_state.result is not None:
    result = st.session_state.result
    st.subheader(f"🎯 Hasil Prediksi Top {top_n} Digit")
    labels = ["As", "Kop", "Kepala", "Ekor"]
    
    for i, label in enumerate(labels):
        hasil_str = ", ".join(map(str, result[i]))
        st.markdown(f"#### **{label}:** `{hasil_str}`")
    
    st.divider()
    # (Sisa kode untuk expander kombinasi dan evaluasi tidak berubah)
    # ...
