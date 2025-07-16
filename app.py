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
    if 'run_putaran_analysis' not in st.session_state:
        st.session_state.run_putaran_analysis = False

init_session_state()

# --- Fungsi Bantuan ---
def analyze_advanced_patterns(historical_df):
    if historical_df is None or historical_df.empty:
        return {}
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
    
    # Gabungkan semua series untuk frekuensi total
    combined_series = pd.concat(series_list)
    combined_freq = combined_series.value_counts()
    
    # Buat frekuensi untuk setiap series individu
    freqs = [s.value_counts().reindex(all_digits, fill_value=0) for s in series_list]

    # Resep 1: Frekuensi Gabungan
    sets.append(combined_freq.nlargest(7).index.tolist())
    
    # Resep 2 & 3: Frekuensi dari 2 posisi pertama
    if len(freqs) > 1:
        sets.append(freqs[0].nlargest(7).index.tolist())
        sets.append(freqs[1].nlargest(7).index.tolist())

    # Resep 4: Gabungan Unik Top
    unique_top = list(dict.fromkeys(freqs[0].nlargest(4).index.tolist() + freqs[1].nlargest(4).index.tolist()))
    for d in combined_freq.nlargest(10).index:
        if len(unique_top) >= 7: break
        if d not in unique_top: unique_top.append(d)
    sets.append(unique_top[:7])
    
    # Resep 5: Frekuensi dengan Bobot
    weighted_counts = sum(freq * (1 + i*0.1) for i, freq in enumerate(freqs))
    sets.append(weighted_counts.nlargest(7).index.tolist())

    # Pastikan semua set memiliki 7 digit
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
    return {
        "2d_depan": get_control_sets_from_series(pos[:2]),
        "2d_tengah": get_control_sets_from_series(pos[1:3]),
        "2d_belakang": get_control_sets_from_series(pos[2:]),
        "4d": get_control_sets_from_series(pos),
    }

def generate_on_off_patterns(result, probs):
    if not result: return {}
    
    # 1. Tentukan 9 digit terkuat untuk 4D ON
    all_predicted_digits = [d for sublist in result for d in sublist]
    on_digits_9 = list(dict.fromkeys(all_predicted_digits))[:9]
    if len(on_digits_9) < 9: # Jika kurang dari 9, tambal dengan angka lain
        sisa = [d for d in range(10) if d not in on_digits_9]
        on_digits_9.extend(sisa[:9-len(on_digits_9)])

    # 2. Hasilkan semua kombinasi dari 9 digit, skor, dan ambil 5000 teratas
    all_on_combos = list(product(on_digits_9, repeat=4))
    scored_combos = []
    for p in all_on_combos:
        score = probs[0][p[0]] + probs[1][p[1]] + probs[2][p[2]] + probs[3][p[3]]
        scored_combos.append(("".join(map(str, p)), score))
    
    scored_combos.sort(key=lambda x: x[1], reverse=True)
    on_4d_5000 = [combo[0] for combo in scored_combos[:5000]]

    # 3. Tentukan digit OFF (yang tidak ada di 9 digit ON)
    off_digit = [d for d in range(10) if d not in on_digits_9]

    return {'on': on_4d_5000, 'off_digit': off_digit[0] if off_digit else '-'}

# ==============================================================================
# --- UI (Tampilan Aplikasi) ---
# ==============================================================================

st.title("📊 Analisis Prediksi 4D")
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

with st.sidebar:
    # ... (kode sidebar tidak berubah) ...
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
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 9, help="Gunakan 9 untuk hasil 4D ON/OFF terbaik")
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
            pass # Kode expander tidak berubah
    
    with col2:
        st.subheader("💡 Pola Lanjutan (Data Historis)")
        patterns = analyze_advanced_patterns(df)
        if patterns:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.text_input("As Off", value=patterns.get('as_off'), disabled=True, key="as_off")
                st.text_input("Kepala Off", value=patterns.get('kepala_off'), disabled=True, key="kep_off")
                st.markdown("---")
                st.text_input("Kembar Depan", value=patterns.get('kembar_depan'), disabled=True, key="kd")
                st.text_input("Kembar Tengah", value=patterns.get('kembar_tengah'), disabled=True, key="kt")
                st.text_input("Kembar Belakang", value=patterns.get('kembar_belakang'), disabled=True, key="kb")
            with sub_col2:
                st.text_input("Kop Off", value=patterns.get('kop_off'), disabled=True, key="kop_off")
                st.text_input("Ekor Off", value=patterns.get('ekor_off'), disabled=True, key="ekor_off")
                st.markdown("---")
                st.text_input("Kembar As-Kepala", value=patterns.get('kembar_as_kep'), disabled=True, key="kak")
                st.text_input("Kembar As-Ekor", value=patterns.get('kembar_as_ekor'), disabled=True, key="kae")
                st.text_input("Kembar Kop-Ekor", value=patterns.get('kembar_kop_ekor'), disabled=True, key="kke")
        
        st.markdown("---")
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
            on_list = patterns_on_off['on']
            st.info(f"Menampilkan 200 dari {len(on_list)} kombinasi.")
            display_text = " ".join(on_list[:200])
            st.text_area("_4d_on_display", value=display_text, height=100, key="4d_on_display")
            full_on_text = "\n".join(on_list)
            st.download_button("Unduh Semua 4D ON", data=full_on_text, file_name="4d_on.txt", use_container_width=True)
    
    with col4d_3:
        st.markdown(f'<p style="background-color:#B22222; color:white; font-weight:bold; text-align:center; padding: 5px; border-radius: 5px 5px 0 0;">4D OFF</p>', unsafe_allow_html=True)
        if patterns_on_off:
            off_digit = patterns_on_off['off_digit']
            st.markdown(f"<div style='padding:10px;'><b>Angka Mati Total:</b><h1 style='text-align: center; color: #B22222;'>{off_digit}</h1></div>", unsafe_allow_html=True)
            st.info("Semua kombinasi yang mengandung angka ini dianggap lemah.")

# ... (Sisa kode untuk Analisis Putaran Terbaik tidak berubah) ...
if st.session_state.get('run_putaran_analysis', False):
    pass
