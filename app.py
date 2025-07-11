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

# Inisialisasi session_state di awal untuk mencegah error
def init_session_state():
    if 'df_data' not in st.session_state:
        st.session_state.df_data = pd.DataFrame()
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    if 'run_putaran_analysis' not in st.session_state:
        st.session_state.run_putaran_analysis = False

init_session_state()

# Fungsi untuk menghitung Colok
def calculate_colok(probabilities):
    if probabilities is None:
        return [], []
    total_probs = np.sum(probabilities, axis=0)
    top_3_cb_digits = np.argsort(total_probs)[-3:][::-1].tolist()
    all_digit_pairs = list(combinations(range(10), 2))
    scored_cm_pairs = [{'pair': pair, 'score': total_probs[pair[0]] + total_probs[pair[1]]} for pair in all_digit_pairs]
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

# --- UI START ---

lottie_predict = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_gfrw22im.json")
if lottie_predict:
    st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("📊 Analisis Prediksi 4D")

metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    data_source = st.radio(
        "Sumber Data", ("API", "Input Manual"), horizontal=True, key='data_source_selector'
    )
    
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
        manual_data_input = st.text_area(
            "📋 Masukkan Data Keluaran", height=150, placeholder="Contoh: 1234 5678, 9012..."
        )
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
            st.session_state.df_data = pd.DataFrame({"angka": angka_list})
            st.session_state.prediction_data = None
            
    st.divider()

    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100)
    metode = st.selectbox("🧠 Metode Analisis", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 8)
    
    st.divider()

    st.header("🔬 Analisis Lanjutan")
    jumlah_uji = st.number_input(
        "📊 Jml Data untuk Back-testing", 1, 200, 10,
        help="Berapa banyak data terakhir yang akan dijadikan 'kunci jawaban' untuk menguji akurasi setiap skenario putaran."
    )
    if st.button("🔍 Analisis Putaran Terbaik"):
        total_data_saat_ini = len(st.session_state.get('df_data', []))
        if total_data_saat_ini < 30:
            st.warning(f"Butuh minimal 30 data riwayat. Saat ini hanya ada **{total_data_saat_ini}** data yang dimuat.")
        else:
            st.session_state.run_putaran_analysis = True
            st.session_state.prediction_data = None

# --- TAMPILAN UTAMA ---
df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)

if not df.empty:
    with st.expander(f"✅ Menampilkan {len(df)} data terakhir yang digunakan.", expanded=False):
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
    st.subheader(f"🎯 Hasil Analisis Top {top_n} Digit")
    labels = ["As", "Kop", "Kepala", "Ekor"]
    for i, label in enumerate(labels):
        hasil_str = ", ".join(map(str, result[i]))
        st.markdown(f"#### **{label}:** `{hasil_str}`")
    st.divider()
    top_cb, top_cm = calculate_colok(probs)
    if top_cb and top_cm:
        cb_str = " ".join(map(str, top_cb))
        cm_str = " ".join(top_cm)
        st.markdown(f"#### **Colok Bebas / CB:** `{cb_str}`")
        st.markdown(f"#### **Makau / CM:** `{cm_str}`")
        st.divider()
    with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
        # ... kode kombinasi ...
        pass


# ==== PERBAIKAN TOTAL PADA LOGIKA ANALISIS PUTARAN TERBAIK ====
if st.session_state.get('run_putaran_analysis', False):
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    with st.spinner("Menganalisis berbagai jumlah putaran..."):
        full_df = st.session_state.df_data
        putaran_results = {}
        
        max_putaran_test = len(full_df) - jumlah_uji
        if max_putaran_test < 20:
             st.warning(f"Data tidak cukup untuk analisis. Butuh total {20 + jumlah_uji} data, hanya ada {len(full_df)}.")
        else:
            test_range = list(range(20, max_putaran_test + 1, 10))
            if max_putaran_test not in test_range:
                test_range.append(max_putaran_test)

            progress_bar = st.progress(0, text="Memulai analisis...")
            
            for i, p in enumerate(test_range):
                total_benar_for_p = 0
                total_digits_for_p = 0

                # Lakukan back-testing untuk setiap nilai p
                for j in range(jumlah_uji):
                    end_index = len(full_df) - jumlah_uji + j
                    start_index = end_index - p
                    
                    if start_index < 0: continue # Pastikan tidak ada index negatif
                    
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
