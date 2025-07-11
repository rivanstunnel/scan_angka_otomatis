# app.py

import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from itertools import product

# Impor fungsi-fungsi dari file model.
from markov_model import (
    predict_markov,
    predict_markov_order2,
    predict_markov_hybrid,
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Prediksi 4D Markov", layout="wide")

# Fungsi untuk mereset status
def reset_data_and_prediction():
    """Hapus data dan hasil prediksi untuk memulai dari awal."""
    st.session_state.df_data = pd.DataFrame()
    st.session_state.result = None
    st.session_state.last_query = ""

def reset_prediction_only():
    """Hanya hapus hasil prediksi jika pengaturan analisis berubah."""
    st.session_state.result = None

# Inisialisasi session_state
if 'df_data' not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if 'result' not in st.session_state:
    st.session_state.result = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

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
    
    data_source = st.radio(
        "Sumber Data", ("API", "Input Manual"), horizontal=True,
        on_change=reset_data_and_prediction, key='data_source_selector'
    )
    
    if data_source == "API":
        hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
        selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list, on_change=reset_data_and_prediction)
        selected_hari = st.selectbox("📅 Pilih Hari", hari_list, on_change=reset_data_and_prediction)
    else:
        manual_data_input = st.text_area(
            "📋 Masukkan Data Keluaran", height=150,
            placeholder="Contoh: 1234 5678, 9012..."
        )
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
            st.session_state.df_data = pd.DataFrame({"angka": angka_list})
            st.session_state.last_query = "manual"
            reset_prediction_only()
            st.rerun()

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100, on_change=reset_prediction_only)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list, on_change=reset_prediction_only)
    top_n = st.number_input("🔢 Jumlah Top Digit Prediksi", 1, 9, 8, on_change=reset_prediction_only)

# --- Logika Pengambilan & Pemrosesan Data ---
if data_source == "API":
    query_id = f"{selected_lokasi}-{selected_hari}"
    if st.session_state.get('last_query') != query_id:
        with st.spinner(f"🔄 Mengambil data untuk {selected_lokasi}..."):
            try:
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                data = response.json()
                all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                st.session_state.df_data = pd.DataFrame({"angka": all_angka})
                st.session_state.last_query = query_id
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal ambil data API: {e}")
                st.session_state.df_data = pd.DataFrame()

df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)

if not df.empty:
    # ==== PERBAIKAN: Mengubah expanded=True menjadi expanded=False ====
    with st.expander(f"✅ Menampilkan {len(df)} data terakhir yang digunakan.", expanded=False):
        st.code("\n".join(df['angka'].tolist()), language="text")

# --- Tombol dan Logika Prediksi ---
if st.button("🔮 Prediksi Sekarang!", use_container_width=True):
    if st.session_state.get('result') is None:
        if len(df) < 11:
            st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
        else:
            with st.spinner("⏳ Melakukan prediksi..."):
                result = None
                if metode == "Markov": result, _ = predict_markov(df, top_n=top_n)
                elif metode == "Markov Order-2": result = predict_markov_order2(df, top_n=top_n)
                elif metode == "Markov Gabungan": result = predict_markov_hybrid(df, top_n=top_n)
                st.session_state.result = result
                st.rerun()
    else:
        st.info("ℹ️ Hasil prediksi sudah ditampilkan. Ubah pengaturan untuk prediksi baru.")

# --- Tampilkan Hasil ---
if st.session_state.get('result') is not None:
    result = st.session_state.result
    st.subheader(f"🎯 Hasil Prediksi Top {top_n} Digit")
    labels = ["As", "Kop", "Kepala", "Ekor"]
    
    for i, label in enumerate(labels):
        hasil_str = ", ".join(map(str, result[i]))
        st.markdown(f"#### **{label}:** `{hasil_str}`")
    
    st.divider()

    with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
        kombinasi_4d_list = ["".join(map(str, p)) for p in product(*result)]
        kombinasi_3d_list = ["".join(map(str, p)) for p in product(*result[1:])]
        kombinasi_2d_list = ["".join(map(str, p)) for p in product(*result[2:])]

        separator = " * "
        text_4d = separator.join(kombinasi_4d_list)
        text_3d = separator.join(kombinasi_3d_list)
        text_2d = separator.join(kombinasi_2d_list)

        tab4d, tab3d, tab2d = st.tabs([f"Kombinasi 4D ({len(kombinasi_4d_list)})", f"Kombinasi 3D ({len(kombinasi_3d_list)})", f"Kombinasi 2D ({len(kombinasi_2d_list)})"])

        with tab4d:
            st.text_area("Hasil 4D (As-Kop-Kepala-Ekor)", text_4d, height=200)
            st.download_button("Unduh 4D.txt", text_4d, file_name="hasil_4d.txt")

        with tab3d:
            st.text_area("Hasil 3D (Kop-Kepala-Ekor)", text_3d, height=200)
            st.download_button("Unduh 3D.txt", text_3d, file_name="hasil_3d.txt")

        with tab2d:
            st.text_area("Hasil 2D (Kepala-Ekor)", text_2d, height=200)
            st.download_button("Unduh 2D.txt", text_2d, file_name="hasil_2d.txt")
