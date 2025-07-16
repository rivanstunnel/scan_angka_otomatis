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
    # Hapus flag lama dan ganti dengan penyimpanan hasil
    if 'run_putaran_analysis' in st.session_state:
        del st.session_state['run_putaran_analysis']
    if 'putaran_results' not in st.session_state:
        st.session_state.putaran_results = None

init_session_state()

# --- Fungsi Bantuan ---
# (Fungsi bantuan lainnya seperti calculate_angka_kontrol, generate_angka_jadi_2d, dll. tetap sama)

def calculate_angka_kontrol(probabilities):
    """
    Menghitung Angka Kontrol berdasarkan matriks probabilitas.
    Panjang digit diatur statis ke 7.
    """
    if probabilities is None or probabilities.shape != (4, 10):
        return {}

    total_probs = np.sum(probabilities, axis=0)
    probs_3d = np.sum(probabilities[1:], axis=0)
    probs_2d = np.sum(probabilities[2:], axis=0)

    ak_global = np.argsort(total_probs)[-7:][::-1].tolist()
    top_3d = np.argsort(probs_3d)[-7:][::-1].tolist()
    top_2d = np.argsort(probs_2d)[-7:][::-1].tolist()

    jagoan_per_posisi = np.argmax(probabilities, axis=1).tolist()
    jagoan_final = list(dict.fromkeys(jagoan_per_posisi))
    
    for digit in ak_global:
        if len(jagoan_final) >= 7:
            break
        if digit not in jagoan_final:
            jagoan_final.append(digit)
            
    if len(jagoan_final) < 7:
        sisa_digit = [d for d in range(10) if d not in jagoan_final]
        needed = 7 - len(jagoan_final)
        jagoan_final.extend(sisa_digit[:needed])

    lemah_global = np.argsort(total_probs)[:2].tolist()

    return {
        "Angka Kontrol (AK)": ak_global,
        "Top 4D (AS-KOP-KEP-EKO)": jagoan_final,
        "Top 3D (KOP-KEP-EKO)": top_3d,
        "Top 2D (KEP-EKO)": top_2d,
        "Angka Lemah (Hindari)": lemah_global,
    }

# --- FUNGSI BARU UNTUK ANALISIS BACK-TESTING ---
def run_backtesting_analysis(full_df, metode, top_n, jumlah_uji, start_putaran=11):
    """
    Menjalankan proses back-testing dan mengembalikan hasilnya dalam bentuk dictionary.
    """
    putaran_results = {}
    max_putaran_test = len(full_df) - jumlah_uji
    end_putaran = max_putaran_test
    
    if end_putaran < start_putaran:
        st.warning(f"Data tidak cukup untuk pengujian. Butuh setidaknya {start_putaran + jumlah_uji} total data riwayat.")
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
            actual_row = full_df.iloc[end_index]
            if len(train_df_for_step) < 11: continue

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
# --- UI (Tampilan Aplikasi) Dimulai di Sini ---
# ==============================================================================

st.title("📊 Analisis Prediksi 4D")

metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    data_source = st.radio("Sumber Data", ("API", "Input Manual"), horizontal=True, key='data_source_selector')
    if data_source == "API":
        # ... (kode API tidak berubah) ...
        hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
        selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
        selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
        if st.button("Muat Data API"):
            with st.spinner(f"🔄 Mengambil data untuk {selected_lokasi}..."):
                try:
                    # ... (logika fetch API) ...
                    st.session_state.df_data = pd.DataFrame({"angka": all_angka})
                    # Reset semua hasil analisis saat data baru dimuat
                    st.session_state.prediction_data = None
                    st.session_state.putaran_results = None
                except Exception as e:
                    # ... (error handling) ...
                    st.session_state.df_data = pd.DataFrame()

    else: # Input Manual
        # ... (kode input manual tidak berubah) ...
        manual_data_input = st.text_area("📋 Masukkan Data Keluaran", height=150, placeholder="Contoh: 1234 5678, 9012...")
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
            st.session_state.df_data = pd.DataFrame({"angka": angka_list})
            # Reset semua hasil analisis saat data baru dimuat
            st.session_state.prediction_data = None
            st.session_state.putaran_results = None

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100)
    metode = st.selectbox("🧠 Metode Analisis", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 7)
    st.divider()
    st.header("🔬 Analisis Lanjutan")
    jumlah_uji = st.number_input("📊 Jml Data untuk Back-testing", 1, 200, 10, help="...")
    
    # --- PERUBAHAN LOGIKA TOMBOL ANALISIS PUTARAN ---
    if st.button("🔍 Analisis Putaran Terbaik"):
        total_data_saat_ini = len(st.session_state.get('df_data', []))
        if total_data_saat_ini < jumlah_uji + 11:
            st.warning(f"Butuh minimal {jumlah_uji + 11} data riwayat untuk analisis ini.")
        else:
            full_df = st.session_state.get('df_data', pd.DataFrame())
            # Panggil fungsi analisis dan simpan hasilnya ke session_state
            st.session_state.putaran_results = run_backtesting_analysis(
                full_df, metode, top_n, jumlah_uji
            )
            # Hapus hasil analisis reguler agar tidak membingungkan
            st.session_state.prediction_data = None


df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)
if not df.empty:
    with st.expander(f"✅ Menampilkan {len(df)} data terakhir...", expanded=False):
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
                # Hapus hasil analisis putaran terbaik jika analisis reguler dijalankan
                st.session_state.putaran_results = None


# --- BAGIAN TAMPILAN UNTUK ANALISIS REGULER ---
if st.session_state.get('prediction_data') is not None:
    # ... (Semua kode untuk menampilkan hasil analisis reguler, angka kontrol, dll. tetap sama) ...
    prediction_data = st.session_state.prediction_data
    result = prediction_data["result"]
    probs = prediction_data["probs"]
    st.subheader(f"🎯 Hasil Analisis Top {top_n} Digit")
    # ... sisa kode tampilan ...


# --- BAGIAN TAMPILAN BARU UNTUK HASIL ANALISIS PUTARAN TERBAIK ---
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

        st.subheader(f"📜 Tabel Hasil Analisis Putaran")
        sorted_chart_data = chart_data.sort_values(by='Akurasi (%)', ascending=False)
        sorted_chart_data['Akurasi (%)'] = sorted_chart_data['Akurasi (%)'].map('{:.2f}%'.format)
        st.dataframe(sorted_chart_data, use_container_width=True)

# Hapus blok 'run_putaran_analysis' yang lama
# if st.session_state.get('run_putaran_analysis', False): ... (BLOK INI SEPENUHNYA DIHAPUS)
