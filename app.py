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
    if 'run_putaran_analysis' not in st.session_state:
        st.session_state.run_putaran_analysis = False

init_session_state()

# --- FUNGSI ANALISIS POLA LANJUTAN YANG DIPERBAIKI ---
def analyze_advanced_patterns(historical_df):
    """
    Menganalisis angka 'off' dan pola kembar berdasarkan frekuensi
    pada data historis yang digunakan.
    """
    if historical_df is None or historical_df.empty:
        return {}

    patterns = {}
    total_rows = len(historical_df)
    data = historical_df['angka'].astype(str).str.zfill(4)

    # 1. Cari angka 'Off' berdasarkan frekuensi kemunculan terendah pada data historis.
    # Digunakan value_counts().idxmin() untuk menemukan angka yang paling jarang muncul.
    patterns['as_off'] = data.str[0].astype(int).value_counts().idxmin()
    patterns['kop_off'] = data.str[1].astype(int).value_counts().idxmin()
    patterns['kepala_off'] = data.str[2].astype(int).value_counts().idxmin()
    patterns['ekor_off'] = data.str[3].astype(int).value_counts().idxmin()

    # 2. Analisis pola 'Kembar' berdasarkan frekuensi historis.
    # Ambang batas 10% (0.10) digunakan sebagai dasar. Jika frekuensi di atas
    # ambang batas, dianggap ada tren (ON).
    threshold = 0.10

    def check_twin_freq(pos1, pos2):
        count = (data.str[pos1] == data.str[pos2]).sum()
        freq = count / total_rows if total_rows > 0 else 0
        return "ON" if freq >= threshold else "OFF"

    patterns['kembar_depan'] = check_twin_freq(0, 1)      # As == Kop
    patterns['kembar_tengah'] = check_twin_freq(1, 2)     # Kop == Kepala
    patterns['kembar_belakang'] = check_twin_freq(2, 3)    # Kepala == Ekor
    patterns['kembar_as_kep'] = check_twin_freq(0, 2)       # As == Kepala
    patterns['kembar_as_ekor'] = check_twin_freq(0, 3)      # As == Ekor
    patterns['kembar_kop_ekor'] = check_twin_freq(1, 3)      # Kop == Ekor

    return patterns

# ==============================================================================
# --- UI (Tampilan Aplikasi) Dimulai di Sini ---
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
            with st.spinner(f"🔄 Mengambil data untuk {selected_lokasi}..."):
                try:
                    url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
                    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                    response = requests.get(url, headers=headers, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                    st.session_state.df_data = pd.DataFrame({"angka": all_angka})
                    st.session_state.prediction_data = None
                except Exception as e:
                    st.error(f"❌ Gagal ambil data API: {e}")
                    st.session_state.df_data = pd.DataFrame()
    else: # Input Manual
        manual_data_input = st.text_area("📋 Masukkan Data Keluaran", height=150, placeholder="Contoh: 1234 5678, 9012...")
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
            st.session_state.df_data = pd.DataFrame({"angka": angka_list})
            st.session_state.prediction_data = None

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100)
    metode = st.selectbox("🧠 Metode Analisis", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 7)
    st.divider()
    st.header("🔬 Analisis Lanjutan")
    jumlah_uji = st.number_input("📊 Jml Data untuk Back-testing", 1, 200, 10, help="...")
    if st.button("🔍 Analisis Putaran Terbaik"):
        total_data_saat_ini = len(st.session_state.get('df_data', []))
        if total_data_saat_ini < jumlah_uji + 11:
            st.warning(f"Butuh minimal {jumlah_uji + 11} data riwayat...")
        else:
            st.session_state.run_putaran_analysis = True
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

if st.session_state.get('prediction_data') is not None:
    prediction_data = st.session_state.prediction_data
    result = prediction_data["result"]
    probs = prediction_data["probs"]

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        st.subheader(f"🎯 Hasil Analisis Top {top_n} Digit")
        labels = ["As", "Kop", "Kepala", "Ekor"]
        for i, label in enumerate(labels):
            hasil_str = ", ".join(map(str, result[i]))
            st.markdown(f"**{label}:** `{hasil_str}`")
        
        st.divider()

        with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
            kombinasi_4d_list = ["".join(map(str, p)) for p in product(*result)]
            kombinasi_3d_list = ["".join(map(str, p)) for p in product(*result[1:])]
            kombinasi_2d_list = ["".join(map(str, p)) for p in product(*result[2:])]
            separator = " * "
            text_4d = separator.join(kombinasi_4d_list)
            text_3d = separator.join(kombinasi_3d_list)
            text_2d = separator.join(kombinasi_2d_list)
            tab2d, tab3d, tab4d = st.tabs([f"2D ({len(kombinasi_2d_list)})", f"3D ({len(kombinasi_3d_list)})", f"4D ({len(kombinasi_4d_list)})"])
            with tab2d: st.text_area("Hasil 2D...", text_2d, height=150, key="txt2d"); st.download_button("Unduh 2D.txt", text_2d)
            with tab3d: st.text_area("Hasil 3D...", text_3d, height=150, key="txt3d"); st.download_button("Unduh 3D.txt", text_3d)
            with tab4d: st.text_area("Hasil 4D...", text_4d, height=150, key="txt4d"); st.download_button("Unduh 4D.txt", text_4d)
    
    with col2:
        st.subheader("💡 Pola Lanjutan (Data Historis)")
        # --- PERBAIKAN: Memanggil fungsi dengan dataframe historis (df) ---
        patterns = analyze_advanced_patterns(df)
        if patterns:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.text_input("As Off", value=patterns.get('as_off'), disabled=True)
                st.text_input("Kepala Off", value=patterns.get('kepala_off'), disabled=True)
                st.markdown("---")
                st.text_input("Kembar Depan", value=patterns.get('kembar_depan'), disabled=True)
                st.text_input("Kembar Tengah", value=patterns.get('kembar_tengah'), disabled=True)
                st.text_input("Kembar Belakang", value=patterns.get('kembar_belakang'), disabled=True)

            with sub_col2:
                st.text_input("Kop Off", value=patterns.get('kop_off'), disabled=True)
                st.text_input("Ekor Off", value=patterns.get('ekor_off'), disabled=True)
                st.markdown("---")
                st.text_input("Kembar As-Kepala", value=patterns.get('kembar_as_kep'), disabled=True)
                st.text_input("Kembar As-Ekor", value=patterns.get('kembar_as_ekor'), disabled=True)
                st.text_input("Kembar Kop-Ekor", value=patterns.get('kembar_kop_ekor'), disabled=True)

    st.divider()

if st.session_state.get('run_putaran_analysis', False):
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    with st.spinner("Menganalisis berbagai jumlah putaran... Ini akan memakan waktu."):
        full_df = st.session_state.get('df_data', pd.DataFrame())
        putaran_results = {}
        max_putaran_test = len(full_df) - jumlah_uji
        start_putaran = 11
        end_putaran = max_putaran_test
        step_putaran = 1

        if end_putaran < start_putaran:
            st.warning(f"Data tidak cukup untuk pengujian. Butuh setidaknya {start_putaran + jumlah_uji} total data riwayat.")
        else:
            test_range = list(range(start_putaran, end_putaran + 1, step_putaran))
            progress_bar = st.progress(0, text="Memulai analisis...")
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
                    if metode == "Markov": 
                        pred, _ = predict_markov(train_df_for_step, top_n=top_n)
                    elif metode == "Markov Order-2": 
                        pred, _ = predict_markov_order2(train_df_for_step, top_n=top_n)
                    elif metode == "Markov Gabungan": 
                        pred, _ = predict_markov_hybrid(train_df_for_step, top_n=top_n)

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

                st.subheader(f"📜 Tabel Hasil Analisis Putaran (Rentang {start_putaran}-{end_putaran})")
                sorted_chart_data = chart_data.sort_values(by='Akurasi (%)', ascending=False)
                sorted_chart_data['Akurasi (%)'] = sorted_chart_data['Akurasi (%)'].map('{:.2f}%'.format)
                st.dataframe(sorted_chart_data, use_container_width=True)

    st.session_state.run_putaran_analysis = False
