# app.py

import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from itertools import product, combinations

# Impor fungsi-fungsi dari file model.
from markov_model import (
    predict_markov,
    predict_markov_order2,
    predict_markov_hybrid,
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Analisis Prediksi 4D", layout="wide")

# Fungsi untuk mereset status
def reset_data_and_prediction():
    st.session_state.df_data = pd.DataFrame()
    st.session_state.prediction_data = None
    st.session_state.last_query = ""

def reset_prediction_only():
    st.session_state.prediction_data = None

# Inisialisasi session_state
if 'df_data' not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# ==== FUNGSI BARU: Untuk menghitung Colok Bebas dan Macau ====
def calculate_colok(probabilities):
    if probabilities is None:
        return [], []
    
    # Hitung total probabilitas untuk setiap digit (0-9) di semua posisi
    total_probs = np.sum(probabilities, axis=0)
    
    # Colok Bebas (CB): Ambil 3 digit dengan total probabilitas tertinggi
    top_3_cb_digits = np.argsort(total_probs)[-3:][::-1]
    
    # Colok Macau (CM): Buat 3 kombinasi pasangan dari 3 digit CB teratas
    cm_pairs = list(combinations(top_3_cb_digits, 2))
    formatted_cm_pairs = [f"{p[0]}{p[1]}" for p in cm_pairs]

    return top_3_cb_digits.tolist(), formatted_cm_pairs


def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None

lottie_predict = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_gfrw22im.json")
if lottie_predict:
    st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("📊 Analisis Prediksi 4D")

metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    data_source = st.radio(
        "Sumber Data", ("API", "Input Manual"), horizontal=True,
        on_change=reset_data_and_prediction, key='data_source_selector'
    )
    if data_source == "API":
        # ... (Tidak ada perubahan di sini) ...
    else:
        # ... (Tidak ada perubahan di sini) ...

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100, on_change=reset_prediction_only)
    metode = st.selectbox("🧠 Metode Analisis", metode_list, on_change=reset_prediction_only)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 8, on_change=reset_prediction_only)

# --- Logika Data ---
# ... (Tidak ada perubahan di sini) ...

# --- Tombol dan Logika Analisis ---
if st.button("📈 Analisis Sekarang!", use_container_width=True):
    if st.session_state.get('prediction_data') is None:
        if len(df) < 11:
            st.warning("❌ Minimal 11 data diperlukan untuk analisis.")
        else:
            with st.spinner("⏳ Melakukan analisis..."):
                result, probs = None, None
                # ==== PERBAIKAN: Menangkap probabilitas dari semua metode ====
                if metode == "Markov": result, probs = predict_markov(df, top_n=top_n)
                elif metode == "Markov Order-2": result, probs = predict_markov_order2(df, top_n=top_n)
                elif metode == "Markov Gabungan": result, probs = predict_markov_hybrid(df, top_n=top_n)
                
                # Simpan hasil dan probabilitas ke session state
                if result is not None:
                    st.session_state.prediction_data = {"result": result, "probs": probs}
                st.rerun()
    else:
        st.info("ℹ️ Hasil analisis sudah ditampilkan. Ubah pengaturan untuk analisis baru.")

# --- Tampilkan Hasil ---
if st.session_state.get('prediction_data') is not None:
    prediction_data = st.session_state.prediction_data
    result = prediction_data["result"]
    probs = prediction_data["probs"]
    
    st.subheader(f"🎯 Hasil Analisis Top {top_n} Digit")
    labels = ["As", "Kop", "Kepala", "Ekor"]
    
    for i, label in enumerate(labels):
        hasil_str = ", ".join(map(str, result[i]))
        st.markdown(f"#### **{label}:** `{hasil_str}`")
    
    st.divider()

    # ==== TAMPILAN BARU: Menampilkan hasil Colok Bebas dan Macau ====
    top_cb, top_cm = calculate_colok(probs)
    if top_cb and top_cm:
        cb_str = " ".join(map(str, top_cb))
        cm_str = " ".join(top_cm)
        st.markdown(f"#### **Colok Bebas / CB:** `{cb_str}`")
        st.markdown(f"#### **Makau / CM:** `{cm_str}`")
        st.divider()

    with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
        # ... (Tidak ada perubahan di sini) ...
