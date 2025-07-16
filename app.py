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

def generate_2d_ai_ct(historical_df):
    if historical_df is None or historical_df.empty:
        return {}
    data = historical_df['angka'].astype(str).str.zfill(4)
    pos_as, pos_kop, pos_kep, pos_ekor = data.str[0].astype(int), data.str[1].astype(int), data.str[2].astype(int), data.str[3].astype(int)
    def get_control_sets(series1, series2):
        all_digits, sets = list(range(10)), []
        combined_freq = pd.concat([series1, series2]).value_counts()
        sets.append(combined_freq.nlargest(7).index.tolist())
        freq1 = series1.value_counts().reindex(all_digits, fill_value=0)
        sets.append(freq1.nlargest(7).index.tolist())
        freq2 = series2.value_counts().reindex(all_digits, fill_value=0)
        sets.append(freq2.nlargest(7).index.tolist())
        top_s1, top_s2 = freq1.nlargest(4).index, freq2.nlargest(4).index
        unique_top = list(dict.fromkeys(top_s1.tolist() + top_s2.tolist()))
        for d in combined_freq.nlargest(10).index:
            if len(unique_top) >= 7: break
            if d not in unique_top: unique_top.append(d)
        sets.append(unique_top[:7])
        weighted_counts = (freq2 * 1.2) + freq1
        sets.append(weighted_counts.nlargest(7).index.tolist())
        final_sets = []
        for s in sets:
            if len(s) < 7:
                s_set = set(s)
                for d in combined_freq.nlargest(10).index:
                    if len(s) >= 7: break
                    if d not in s_set: s.append(d)
            final_sets.append(s[:7])
        return final_sets
    return {"depan": get_control_sets(pos_as, pos_kop), "tengah": get_control_sets(pos_kop, pos_kep), "belakang": get_control_sets(pos_kep, pos_ekor)}

# --- FUNGSI BARU UNTUK MENGHASILKAN POLA 4D ---
def generate_4d_patterns(result):
    """
    Menghasilkan AI/CT 4D, 4D ON, dan 4D OFF dari hasil prediksi.
    """
    if not result:
        return {}

    # 1. Buat AI/CT 4D (7 digit) dari digit-digit terkuat
    top_digits = []
    for i in range(4):
        top_digits.extend(result[i][:2])  # Ambil 2 teratas dari setiap posisi
    
    ai_ct_4d = list(dict.fromkeys(top_digits)) # Ambil digit unik
    
    # Tambal hingga 7 digit jika kurang, menggunakan sisa prediksi
    all_predicted_digits = [d for sublist in result for d in sublist]
    for digit in all_predicted_digits:
        if len(ai_ct_4d) >= 7: break
        if digit not in ai_ct_4d: ai_ct_4d.append(digit)
    ai_ct_4d = ai_ct_4d[:7]

    # 2. Hasilkan 4D ON dari kombinasi AI/CT 4D
    on_4d = ["".join(map(str, p)) for p in product(ai_ct_4d, repeat=4)]

    # 3. Hasilkan 4D OFF dari digit terlemah (yang tidak ada di AI/CT 4D)
    off_digits = [d for d in range(10) if d not in ai_ct_4d]
    off_4d = ["".join(map(str, p)) for p in product(off_digits, repeat=4)] if off_digits else []

    return {'ai_ct': ai_ct_4d, 'on': on_4d, 'off': off_4d}


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
            # ... (kode expander tidak berubah) ...
            pass
    
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
        
        st.markdown("---", help="Pemisah Bagian")
        ai_ct_results = generate_2d_ai_ct(df)
        if ai_ct_results:
            ct1, ct2, ct3 = st.columns(3)
            def display_card(column, title, data_key):
                with column:
                    st.markdown(f'<p style="background-color:#B22222; color:white; font-weight:bold; text-align:center; padding: 5px; border-radius: 5px 5px 0 0;">{title}</p>', unsafe_allow_html=True)
                    text_content = "\n".join(["".join(map(str, row)) for row in ai_ct_results.get(data_key, [])])
                    st.text_area(label=f"_{title}", value=text_content, height=140, key=f"ct_{data_key}", label_visibility="collapsed")
            display_card(ct1, "AI/CT 2D Depan", "depan")
            display_card(ct2, "AI/CT 2D Tengah", "tengah")
            display_card(ct3, "AI/CT 2D Belakang", "belakang")
        
        # --- BAGIAN BARU UNTUK POLA 4D ---
        st.markdown("---")
        st.subheader("Pola 4D (Berdasarkan Prediksi)")
        patterns_4d = generate_4d_patterns(result)
        if patterns_4d:
            p4d_c1, p4d_c2, p4d_c3 = st.columns(3)
            
            def display_4d_card(column, title, color, data, key):
                with column:
                    st.markdown(f'<p style="background-color:{color}; color:white; font-weight:bold; text-align:center; padding: 5px; border-radius: 5px 5px 0 0;">{title}</p>', unsafe_allow_html=True)
                    text_content = " ".join(map(str, data))
                    st.text_area(f"_{title}", value=text_content, height=150, key=key, label_visibility="collapsed")

            display_4d_card(p4d_c1, "AI/CT 4D", "#4682B4", patterns_4d['ai_ct'], "ai_ct_4d")
            display_4d_card(p4d_c2, "4D ON", "#2E8B57", patterns_4d['on'], "4d_on")
            display_4d_card(p4d_c3, "4D OFF", "#B22222", patterns_4d['off'], "4d_off")

    st.divider()

# ... (Sisa kode untuk Analisis Putaran Terbaik tidak berubah) ...
if st.session_state.get('run_putaran_analysis', False):
    #...
    pass
