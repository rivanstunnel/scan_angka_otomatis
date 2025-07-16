# app.py

import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from itertools import product

# Impor dari file model Anda
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
def analyze_advanced_patterns(historical_df):
    if historical_df is None or historical_df.empty: return {}
    patterns = {}
    total_rows = len(historical_df)
    data = historical_df['angka'].astype(str).str.zfill(4)
    patterns['as_off'] = data.str[0].astype(int).value_counts().idxmin()
    patterns['kop_off'] = data.str[1].astype(int).value_counts().idxmin()
    patterns['kepala_off'] = data.str[2].astype(int).value_counts().idxmin()
    patterns['ekor_off'] = data.str[3].astype(int).value_counts().idxmin()
    threshold = 0.10
    def check_twin_freq(pos1, pos2):
        count = (data.str[pos1] == data.str[pos2]).sum()
        freq = count / total_rows if total_rows > 0 else 0
        return "ON" if freq >= threshold else "OFF"
    patterns['kembar_depan'] = check_twin_freq(0, 1)
    patterns['kembar_tengah'] = check_twin_freq(1, 2)
    patterns['kembar_belakang'] = check_twin_freq(2, 3)
    patterns['kembar_as_kep'] = check_twin_freq(0, 2)
    patterns['kembar_as_ekor'] = check_twin_freq(0, 3)
    patterns['kembar_kop_ekor'] = check_twin_freq(1, 3)
    return patterns

def get_control_sets_from_series(series_list):
    all_digits, sets = list(range(10)), []
    combined_series = pd.concat(series_list)
    combined_freq = combined_series.value_counts()
    freqs = [s.value_counts().reindex(all_digits, fill_value=0) for s in series_list]
    sets.append(combined_freq.nlargest(7).index.tolist())
    if len(freqs) > 1:
        sets.append(freqs[0].nlargest(7).index.tolist())
        sets.append(freqs[1].nlargest(7).index.tolist())
    unique_top = list(dict.fromkeys(freqs[0].nlargest(4).index.tolist() + freqs[1].nlargest(4).index.tolist()))
    for d in combined_freq.nlargest(10).index:
        if len(unique_top) >= 7: break
        if d not in unique_top: unique_top.append(d)
    sets.append(unique_top[:7])
    weighted_counts = sum(freq * (1 + i*0.1) for i, freq in enumerate(freqs))
    sets.append(weighted_counts.nlargest(7).index.tolist())
    final_sets = []
    for s in sets:
        if len(s) < 7:
            s_set = set(s)
            for d in combined_freq.nlargest(10).index:
                if len(s) >= 7: break
                if d not in s_set: s.append(d)
        final_sets.append(s[:7])
    while len(final_sets) < 5:
        final_sets.append(final_sets[0])
    return final_sets[:5]

def generate_ai_ct_patterns(historical_df):
    if historical_df is None or historical_df.empty: return {}
    data = historical_df['angka'].astype(str).str.zfill(4)
    pos = [data.str[i].astype(int) for i in range(4)]
    return { "2d_depan": get_control_sets_from_series(pos[:2]), "2d_tengah": get_control_sets_from_series(pos[1:3]), "2d_belakang": get_control_sets_from_series(pos[2:]), "4d": get_control_sets_from_series(pos) }

def generate_on_off_patterns(result, probs):
    if not result: return {}
    all_predicted_digits = [d for sublist in result for d in sublist]
    on_digits_9 = list(dict.fromkeys(all_predicted_digits))[:9]
    if len(on_digits_9) < 9:
        sisa = [d for d in range(10) if d not in on_digits_9]
        on_digits_9.extend(sisa[:9-len(on_digits_9)])
    all_on_combos = list(product(on_digits_9, repeat=4))
    scored_combos = []
    for p in all_on_combos:
        score = probs[0][p[0]] + probs[1][p[1]] + probs[2][p[2]] + probs[3][p[3]]
        scored_combos.append(("".join(map(str, p)), score))
    scored_combos.sort(key=lambda x: x[1], reverse=True)
    on_4d_5000 = [combo[0] for combo in scored_combos[:5000]]
    all_4d_numbers = {f"{i:04d}" for i in range(10000)}
    on_4d_set = set(on_4d_5000)
    off_4d_set = all_4d_numbers - on_4d_set
    off_4d_list = sorted(list(off_4d_set))
    return {'on': on_4d_5000, 'off': off_4d_list}

def run_backtesting_analysis(full_df, metode, top_n, jumlah_uji):
    putaran_results = {}
    max_putaran_test = len(full_df) - jumlah_uji
    start_putaran = 11
    end_putaran = max(start_putaran, max_putaran_test)
    if end_putaran < start_putaran: return {}
    test_range = list(range(start_putaran, end_putaran + 1))
    progress_bar = st.progress(0, text="Memulai analisis...")
    for i, p in enumerate(test_range):
        total_benar_for_p, total_digits_for_p = 0, 0
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
                    if int(actual_digits[k]) in pred[k]: total_benar_for_p += 1
                total_digits_for_p += 4
        accuracy = (total_benar_for_p / total_digits_for_p * 100) if total_digits_for_p > 0 else 0
        if accuracy > 0: putaran_results[p] = accuracy
        progress_bar.progress((i + 1) / len(test_range), text=f"Menganalisis {p} putaran...")
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
    def load_data(df_new):
        st.session_state.df_data = df_new
        st.session_state.prediction_data = None
        st.session_state.putaran_results = None
    if data_source == "API":
        hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
        selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list, key="sb_lokasi")
        selected_hari = st.selectbox("📅 Pilih Hari", hari_list, key="sb_hari")
        if st.button("Muat Data API", key="btn_muat_api"):
            with st.spinner(f"🔄 Mengambil data untuk {selected_lokasi}..."):
                try:
                    url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
                    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                    response = requests.get(url, headers=headers, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                    load_data(pd.DataFrame({"angka": all_angka}))
                    st.success(f"Berhasil memuat {len(all_angka)} data.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Gagal mengambil data: Masalah jaringan atau API. Error: {e}")
                except Exception as e:
                    st.error(f"Terjadi kesalahan tidak terduga: {e}")
                    load_data(pd.DataFrame())
    else: # Input Manual
        manual_data_input = st.text_area("📋 Masukkan Data Keluaran", height=150, placeholder="Contoh: 1234 5678, 9012...")
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
            load_data(pd.DataFrame({"angka": angka_list}))
            st.success(f"Berhasil memproses {len(angka_list)} data.")
    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100)
    metode = st.selectbox("🧠 Metode Analisis", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 9, help="Gunakan 9 untuk hasil 4D ON/OFF terbaik")
    st.divider()
    st.header("🔬 Analisis Lanjutan")
    jumlah_uji = st.number_input("📊 Jml Data untuk Back-testing", 1, 200, 1)
    if st.button("🔍 Analisis Putaran Terbaik"):
        total_data_saat_ini = len(st.session_state.get('df_data', []))
        if total_data_saat_ini < jumlah_uji + 11:
            st.warning(f"Butuh minimal {jumlah_uji + 11} data riwayat.")
        else:
            full_df = st.session_state.get('df_data', pd.DataFrame())
            with st.spinner("Menjalankan Analisis Putaran... Ini mungkin memakan waktu."):
                st.session_state.putaran_results = run_backtesting_analysis(full_df, metode, top_n, jumlah_uji)
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
                st.session_state.putaran_results = None

# --- TAMPILAN HASIL ANALISIS REGULER ---
if st.session_state.get('prediction_data') is not None:
    prediction_data = st.session_state.prediction_data
    result = prediction_data["result"]
    probs = prediction_data["probs"]

    st.subheader(f"🎯 Hasil Analisis Top {top_n} Digit")
    labels = ["As", "Kop", "Kepala", "Ekor"]
    top_cols = st.columns(4)
    for i, label in enumerate(labels):
        with top_cols[i]:
            hasil_str = ", ".join(map(str, result[i]))
            st.markdown(f"**{label}:** `{hasil_str}`")
    
    with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
        kombinasi_4d_list = ["".join(map(str, p)) for p in product(*result)]
        kombinasi_3d_list = ["".join(map(str, p)) for p in product(*result[1:])]
        kombinasi_2d_list = ["".join(map(str, p)) for p in product(*result[2:])]
        separator = " * "
        text_4d, text_3d, text_2d = separator.join(kombinasi_4d_list), separator.join(kombinasi_3d_list), separator.join(kombinasi_2d_list)
        tab2d, tab3d, tab4d = st.tabs([f"2D ({len(kombinasi_2d_list)})", f"3D ({len(kombinasi_3d_list)})", f"4D ({len(kombinasi_4d_list)})"])
        with tab2d: st.text_area("Hasil 2D...", text_2d, height=150, key="txt2d"); st.download_button("Unduh 2D.txt", text_2d, key="dl2d")
        with tab3d: st.text_area("Hasil 3D...", text_3d, height=150, key="txt3d"); st.download_button("Unduh 3D.txt", text_3d, key="dl3d")
        with tab4d: st.text_area("Hasil 4D...", text_4d, height=150, key="txt4d"); st.download_button("Unduh 4D.txt", text_4d, key="dl4d")

    st.divider()

    # --- PERUBAHAN TATA LETAK ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💡 Pola Lanjutan (Data Historis)")
        patterns = analyze_advanced_patterns(df)
        if patterns:
            # Bagian Angka Off
            st.text_input("As Off", value=patterns.get('as_off'), disabled=True, key="as_off")
            st.text_input("Kop Off", value=patterns.get('kop_off'), disabled=True, key="kop_off")
            st.text_input("Kepala Off", value=patterns.get('kepala_off'), disabled=True, key="kep_off")
            st.text_input("Ekor Off", value=patterns.get('ekor_off'), disabled=True, key="ekor_off")
            
            st.markdown("---")
            
            # Bagian Pola Kembar dalam 2 kolom
            kembar_col1, kembar_col2 = st.columns(2)
            with kembar_col1:
                st.text_input("Kembar Depan", value=patterns.get('kembar_depan'), disabled=True, key="kd")
                st.text_input("Kembar Tengah", value=patterns.get('kembar_tengah'), disabled=True, key="kt")
                st.text_input("Kembar Belakang", value=patterns.get('kembar_belakang'), disabled=True, key="kb")
            with kembar_col2:
                st.text_input("Kembar As-Kepala", value=patterns.get('kembar_as_kep'), disabled=True, key="kak")
                st.text_input("Kembar As-Ekor", value=patterns.get('kembar_as_ekor'), disabled=True, key="kae")
                st.text_input("Kembar Kop-Ekor", value=patterns.get('kembar_kop_ekor'), disabled=True, key="kke")

    with col2:
        st.subheader("AI/CT Berdasarkan Histori")
        ai_ct_results = generate_ai_ct_patterns(df)
        if ai_ct_results:
            ct1, ct2, ct3 = st.columns(3)
            def display_card(column, title, data_key, main_data):
                with column:
                    st.markdown(f'<p style="background-color:#B22222; color:white; font-weight:bold; text-align:center; padding: 5px; border-radius: 5px 5px 0 0;">{title}</p>', unsafe_allow_html=True)
                    text_content = "\n".join(["".join(map(str, row)) for row in main_data.get(data_key, [])])
                    st.text_area(label=f"_{title}", value=text_content, height=140, key=f"ct_{data_key}", label_visibility="collapsed")
            display_card(ct1, "AI/CT 2D Depan", "2d_depan", ai_ct_results)
            display_card(ct2, "AI/CT 2D Tengah", "2d_tengah", ai_ct_results)
            display_card(ct3, "AI/CT 2D Belakang", "2d_belakang", ai_ct_results)

    st.divider()
    st.subheader("Pola 4D Lengkap (Berdasarkan Prediksi)")
    st.info("Catatan: Menampilkan daftar 4D ON/OFF secara penuh dapat memperlambat kinerja aplikasi.")
    
    col4d_1, col4d_2, col4d_3 = st.columns(3)
    with col4d_1:
        st.markdown(f'<p style="background-color:#4682B4; color:white; font-weight:bold; text-align:center; padding: 5px; border-radius: 5px 5px 0 0;">AI/CT 4D</p>', unsafe_allow_html=True)
        if ai_ct_results:
            text_content_4d = "\n".join(["".join(map(str, row)) for row in ai_ct_results.get("4d", [])])
            st.text_area(label="_ai_ct_4d", value=text_content_4d, height=180, key="ai_ct_4d", label_visibility="collapsed")
    patterns_on_off = generate_on_off_patterns(result, probs)
    with col4d_2:
        st.markdown(f'<p style="background-color:#2E8B57; color:white; font-weight:bold; text-align:center; padding: 5px; border-radius: 5px 5px 0 0;">4D ON</p>', unsafe_allow_html=True)
        if patterns_on_off:
            st.text_area("_4d_on_display", value=" * ".join(patterns_on_off['on']), height=180, key="4d_on_display")
    with col4d_3:
        st.markdown(f'<p style="background-color:#B22222; color:white; font-weight:bold; text-align:center; padding: 5px; border-radius: 5px 5px 0 0;">4D OFF</p>', unsafe_allow_html=True)
        if patterns_on_off:
            st.text_area("_4d_off_display", value=" * ".join(patterns_on_off['off']), height=180, key="4d_off_display")

# TAMPILAN HASIL ANALISIS PUTARAN TERBAIK
if st.session_state.get('putaran_results') is not None:
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    putaran_results = st.session_state.get('putaran_results')
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
