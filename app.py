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

st.set_page_config(page_title="Analisis Prediksi 4D", layout="wide")

# --- Inisialisasi State Aplikasi ---
def init_session_state():
    if 'df_data' not in st.session_state:
        st.session_state.df_data = pd.DataFrame()
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    if 'putaran_results' not in st.session_state:
        st.session_state.putaran_results = None

init_session_state()

# --- Fungsi Bantuan ---
# (Fungsi bantuan lainnya seperti calculate_angka_kontrol, dll. tetap sama)
def calculate_angka_kontrol(probabilities):
    if probabilities is None or probabilities.shape != (4, 10): return {}
    total_probs = np.sum(probabilities, axis=0)
    ak_global = np.argsort(total_probs)[-7:][::-1].tolist()
    lemah_global = np.argsort(total_probs)[:2].tolist()
    return {"Angka Kontrol (AK)": ak_global, "Angka Lemah (Hindari)": lemah_global}

def run_backtesting_analysis(full_df, metode, top_n, jumlah_uji, start_putaran=11):
    putaran_results = {}
    max_putaran_test = len(full_df) - jumlah_uji
    end_putaran = max_putaran_test
    
    if end_putaran < start_putaran:
        st.warning(f"Data tidak cukup. Butuh setidaknya {start_putaran + jumlah_uji} data riwayat.")
        return None

    test_range = list(range(start_putaran, end_putaran + 1, 1))
    progress_bar = st.progress(0, text="Memulai analisis back-testing...")
    
    for i, p in enumerate(test_range):
        total_benar_for_p = 0
        total_digits_for_p = 0
        for j in range(jumlah_uji):
            end_index = len(full_df) - jumlah_uji + j
            start_index = end_index - p
            if start_index < 0: continue
            
            train_df_for_step = full_df.iloc[start_index:end_index]
            if len(train_df_for_step) < 11: continue
            
            actual_row = full_df.iloc[end_index]
            pred, _ = None, None
            if metode == "Markov": pred, _ = predict_markov(train_df_for_step, top_n=top_n)
            elif metode == "Markov Order-2": pred, _ = predict_markov_order2(train_df_for_step, top_n=top_n)
            elif metode == "Markov Gabungan": pred, _ = predict_markov_hybrid(train_df_for_step, top_n=top_n)

            if pred is not None:
                actual_digits = f"{int(actual_row['angka']):04d}"
                for k in range(4):
                    if int(actual_digits[k]) in pred[k]:
                        total_benar_for_p += 1
                total_digits_for_p += 4

        accuracy = (total_benar_for_p / total_digits_for_p * 100) if total_digits_for_p > 0 else 0
        if accuracy > 0:
            putaran_results[p] = accuracy
        
        progress_text = f"Menganalisis {p} putaran... ({i+1}/{len(test_range)})"
        progress_bar.progress((i + 1) / len(test_range), text=progress_text)

    progress_bar.empty()
    return putaran_results

# ==============================================================================
# --- UI (Tampilan Aplikasi) ---
# ==============================================================================

st.title("📊 Analisis Prediksi 4D")

metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    data_source = st.radio("Sumber Data", ("API", "Input Manual"), horizontal=True, key='data_source_selector')
    
    if data_source == "API":
        hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
        selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
        selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
        
        if st.button("Muat Data API"):
            with st.spinner(f"🔄 Menghubungi API untuk {selected_lokasi}..."):
                try:
                    url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
                    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                    response = requests.get(url, headers=headers, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                    st.session_state.df_data = pd.DataFrame({"angka": all_angka})
                    st.session_state.prediction_data = None
                    st.session_state.putaran_results = None
                    st.success(f"Berhasil memuat {len(all_angka)} data.")
                # --- PERBAIKAN: Penanganan Error Lebih Spesifik ---
                except requests.exceptions.Timeout:
                    st.error("Gagal mengambil data: Waktu koneksi habis. Coba lagi nanti.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Gagal mengambil data: Masalah jaringan atau API. Error: {e}")
                except Exception as e:
                    st.error(f"Terjadi kesalahan tidak terduga: {e}")

    else: # Input Manual
        manual_data_input = st.text_area("📋 Masukkan Data Keluaran", height=150, placeholder="Contoh: 1234 5678, 9012...")
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
            st.session_state.df_data = pd.DataFrame({"angka": angka_list})
            st.session_state.prediction_data = None
            st.session_state.putaran_results = None
            st.success(f"Berhasil memproses {len(angka_list)} data.")

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100)
    metode = st.selectbox("🧠 Metode Analisis", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 7)
    st.divider()
    st.header("🔬 Analisis Lanjutan")
    jumlah_uji = st.number_input("📊 Jml Data untuk Back-testing", 1, 200, 10, help="Jumlah data yang dijadikan target pengujian untuk setiap skenario putaran.")
    
    # --- PERBAIKAN: Menambahkan Peringatan Dinamis ---
    total_data_saat_ini = len(st.session_state.get('df_data', []))
    if total_data_saat_ini > 0:
        # Perkirakan jumlah skenario yang akan diuji
        num_putaran_to_test = max(0, total_data_saat_ini - jumlah_uji - 11)
        if num_putaran_to_test > 50: # Tampilkan peringatan jika pengujiannya besar
             st.info(f"ℹ️ Analisis akan menguji ~{num_putaran_to_test} skenario. Proses ini mungkin akan berjalan lambat dan aplikasi tidak akan merespons selama beberapa saat.")

    if st.button("🔍 Analisis Putaran Terbaik"):
        if total_data_saat_ini < jumlah_uji + 11:
            st.warning(f"Butuh minimal {jumlah_uji + 11} data riwayat untuk analisis ini.")
        else:
            full_df = st.session_state.get('df_data', pd.DataFrame())
            st.session_state.putaran_results = run_backtesting_analysis(full_df, metode, top_n, jumlah_uji)
            st.session_state.prediction_data = None

# --- Bagian Utama untuk Menampilkan Hasil ---
df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)
if not df.empty:
    with st.expander(f"✅ Menampilkan {len(df)} dari {len(st.session_state.df_data)} data terakhir (paling baru: {df['angka'].iloc[-1]})", expanded=False):
        st.code("\n".join(df['angka'].tolist()), language="text")

if st.button("📈 Analisis Sekarang!", use_container_width=True):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk analisis.")
    else:
        with st.spinner("⏳ Melakukan analisis..."):
            result, probs = None, None
            if metode == "Markov": result, probs = predict_markov(df, top_n=top_n)
            elif metode == "Markov Order-2": result, probs = predict_markov_order2(df, top_n=top_n)
            elif metode == "Markov Gabungan": result, probs = predict_markov_hybrid(df, top_n=top_n)
            if result is not None:
                st.session_state.prediction_data = {"result": result, "probs": probs}
                st.session_state.putaran_results = None

# --- Tampilkan Hasil Analisis Reguler ---
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

# --- Tampilkan Hasil Analisis Putaran Terbaik ---
if st.session_state.get('putaran_results') is not None:
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    putaran_results = st.session_state.putaran_results
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
        chart_data.index.name = 'Jumlah Putaran'
        st.line_chart(chart_data)
