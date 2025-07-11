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
    # ==== MENU BARU 1: Input untuk data uji ====
    jumlah_uji = st.number_input("📊 Data Uji Analisis Putaran", 1, 200, 10)

    # ==== MENU BARU 2: Tombol untuk memulai analisis putaran ====
    if st.button("🔍 Analisis Putaran Terbaik"):
        if 'df_data' not in st.session_state or len(st.session_state.df_data) < 30:
            st.warning("Butuh minimal 30 data untuk fitur ini.")
        else:
            st.session_state.run_putaran_analysis = True
            reset_prediction_only() # Hapus hasil prediksi lama

# --- (Sisa kode sebagian besar tetap sama) ---
# ... (Logika Pengambilan Data) ...
# ... (Tombol Analisis Sekarang!) ...
# ... (Tampilkan Hasil) ...

# ==== LOGIKA BARU: Menjalankan analisis putaran jika tombol ditekan ====
if st.session_state.get('run_putaran_analysis', False):
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    with st.spinner("Menganalisis berbagai jumlah putaran, ini mungkin memakan waktu..."):
        full_df = st.session_state.df_data
        putaran_results = {}
        
        # Tentukan rentang putaran yang akan diuji
        max_putaran = len(full_df) - jumlah_uji
        test_range = list(range(20, max_putaran, 10))
        if max_putaran not in test_range and max_putaran > 20:
            test_range.append(max_putaran)

        progress_bar = st.progress(0, text="Memulai analisis...")
        
        # Loop untuk setiap nilai putaran
        for i, p in enumerate(test_range):
            df_slice = full_df.tail(p + jumlah_uji)
            uji_df_slice = df_slice.tail(jumlah_uji)
            train_df_slice = df_slice.head(p)

            total, benar = 0, 0
            
            # Lakukan back-testing
            if len(uji_df_slice) > 0:
                pred, _ = None, None
                if metode == "Markov": pred, _ = predict_markov(train_df_slice, top_n=top_n)
                elif metode == "Markov Order-2": pred, _ = predict_markov_order2(train_df_slice, top_n=top_n)
                elif metode == "Markov Gabungan": pred, _ = predict_markov_hybrid(train_df_slice, top_n=top_n)
                
                if pred is not None:
                    for _, row in uji_df_slice.iterrows():
                        actual = f"{int(row['angka']):04d}"
                        for k in range(4):
                            if int(actual[k]) in pred[k]:
                                benar += 1
                        total += 4
            
            accuracy = (benar / total * 100) if total > 0 else 0
            if accuracy > 0:
                putaran_results[p] = accuracy
            progress_bar.progress((i + 1) / len(test_range), text=f"Menganalisis {p} putaran...")
        
        progress_bar.empty()

    if not putaran_results:
        st.error("Tidak dapat menemukan hasil akurasi. Coba dengan metode atau data yang berbeda.")
    else:
        best_putaran = max(putaran_results, key=putaran_results.get)
        best_accuracy = putaran_results[best_putaran]
        
        st.subheader("🏆 Rekomendasi Penggunaan Data")
        m1, m2 = st.columns(2)
        m1.metric("Putaran Terbaik", f"{best_putaran} Data", "Jumlah data historis")
        m2.metric("Akurasi Tertinggi", f"{best_accuracy:.2f}%", f"Dengan {best_putaran} data")
        
        chart_data = pd.DataFrame.from_dict(putaran_results, orient='index', columns=['Akurasi (%)'])
        chart_data.index.name = 'Jumlah Putaran Digunakan'
        st.line_chart(chart_data)
    
    st.session_state.run_putaran_analysis = False
