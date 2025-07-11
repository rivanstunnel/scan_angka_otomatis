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
    st.session_state.run_putaran_analysis = False


def reset_prediction_only():
    st.session_state.prediction_data = None
    st.session_state.run_putaran_analysis = False

# Inisialisasi session_state
if 'df_data' not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'run_putaran_analysis' not in st.session_state:
    st.session_state.run_putaran_analysis = False

# Fungsi baru untuk menghitung Colok
def calculate_colok(probabilities):
    if probabilities is None:
        return [], []
    total_probs = np.sum(probabilities, axis=0)
    top_3_cb_digits = np.argsort(total_probs)[-3:][::-1].tolist()
    all_digit_pairs = list(combinations(range(10), 2))
    scored_cm_pairs = []
    for pair in all_digit_pairs:
        score = total_probs[pair[0]] + total_probs[pair[1]]
        scored_cm_pairs.append({'pair': pair, 'score': score})
    top_3_cm = sorted(scored_cm_pairs, key=lambda x: x['score'], reverse=True)[:3]
    formatted_cm_pairs = [f"{item['pair'][0]}{item['pair'][1]}" for item in top_3_cm]
    return top_3_cb_digits, formatted_cm_pairs

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
    metode = st.selectbox("🧠 Metode Analisis", metode_list, on_change=reset_prediction_only)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 8, on_change=reset_prediction_only)
    
    st.divider()
    st.header("🔬 Analisis Lanjutan")
    
    # ==== PERBAIKAN 1: Label input lebih jelas dengan tooltip (bantuan) ====
    jumlah_uji = st.number_input(
        "📊 Jml Data untuk Back-testing", 1, 200, 10,
        help="Berapa banyak data terakhir yang akan dijadikan 'kunci jawaban' untuk menguji akurasi setiap skenario putaran. Contoh: jika 10, maka 10 data terakhir akan diuji."
    )

    if st.button("🔍 Analisis Putaran Terbaik"):
        # ==== PERBAIKAN 2: Pesan peringatan lebih informatif ====
        total_data_saat_ini = len(st.session_state.get('df_data', []))
        if total_data_saat_ini < 30:
            st.warning(f"Butuh minimal 30 data riwayat. Saat ini hanya ada **{total_data_saat_ini}** data yang dimuat.")
        else:
            st.session_state.run_putaran_analysis = True
            reset_prediction_only() # Hapus hasil prediksi lama

# --- (Sisa kode tidak ada perubahan) ---
# ...
