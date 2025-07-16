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

def calculate_new_controls(probabilities, last_result):
    """
    Menghitung Angka Kontrol, Mati, Shio, dan Jarak Lemah.
    """
    if probabilities is None or probabilities.shape != (4, 10):
        return {}

    total_probs = np.sum(probabilities, axis=0)

    # 1. Angka Kontrol (AK): 7 digit terkuat
    angka_kontrol = np.argsort(total_probs)[-7:][::-1].tolist()

    # 2. Angka Mati: 2 digit terlemah
    angka_mati = np.argsort(total_probs)[:2].tolist()

    # 3. Angka Shio: Kombinasi dari 2 Kepala dan 2 Ekor terkuat
    top_kepala = np.argsort(probabilities[2])[-2:][::-1]
    top_ekor = np.argsort(probabilities[3])[-2:][::-1]
    shio_combinations = list(product(top_kepala, top_ekor))
    angka_shio = sorted(["".join(map(str, p)) for p in shio_combinations])
    
    # 4. Jarak Lemah: Berdasarkan 2D terakhir
    jarak_lemah = "-"
    if last_result and len(last_result) == 4 and last_result.isdigit():
        ekor_2d = int(last_result[2:])
        lemah_start = (ekor_2d + 1) % 100
        lemah_end = (ekor_2d + 20) % 100
        jarak_lemah = f"{lemah_start:02d} s/d {lemah_end:02d}"

    return {
        "Angka Kontrol": angka_kontrol,
        "Angka Mati": angka_mati,
        "Angka Shio": angka_shio,
        "Jarak Lemah": jarak_lemah,
    }

def generate_angka_jadi_2d(probabilities, bbfs_digits):
    if probabilities is None or not bbfs_digits: return []
    all_2d_lines = list(product(bbfs_digits, repeat=2))
    scored_lines = []
    for line in all_2d_lines:
        kepala, ekor = int(line[0]), int(line[1])
        # Pengecekan batas indeks untuk menghindari error
        if kepala < probabilities.shape[1] and ekor < probabilities.shape[1]:
            score = probabilities[2][kepala] + probabilities[3][ekor]
            scored_lines.append(("".join(map(str, line)), score))
    sorted_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)
    return [line[0] for line in sorted_lines]

def generate_angka_jadi_4d(probabilities, bbfs_source_digits):
    if probabilities is None or not bbfs_source_digits: return []
    all_4d_lines = list(product(bbfs_source_digits, repeat=4))
    scored_lines = []
    for line in all_4d_lines:
        a, k, p, e = map(int, line)
        # Pengecekan batas indeks untuk menghindari error
        if a < probabilities.shape[1] and k < probabilities.shape[1] and \
           p < probabilities.shape[1] and e < probabilities.shape[1]:
            score = probabilities[0][a] + probabilities[1][k] + probabilities[2][p] + probabilities[3][e]
            scored_lines.append(("".join(map(str, line)), score))
    sorted_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)
    return [line[0] for line in sorted_lines]

# ==============================================================================
# --- UI (Tampilan Aplikasi) Dimulai di Sini ---
# ==============================================================================

st.title("📊 Analisis Prediksi 4D")

metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

# (Kode sidebar tetap sama, tidak perlu diubah)
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
    with st.expander(f"✅ Menampilkan {len(df)} data terakhir (paling baru: {df['angka'].iloc[-1]})", expanded=False):
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

    # (Expander untuk kombinasi tetap sama)
    with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
        kombinasi_4d_list = ["".join(map(str, p)) for p in product(*result)]
        kombinasi_3d_list = ["".join(map(str, p)) for p in product(*result[1:])]
        kombinasi_2d_list = ["".join(map(str, p)) for p in product(*result[2:])]
        separator = " * "
        text_4d = separator.join(kombinasi_4d_list)
        text_3d = separator.join(kombinasi_3d_list)
        text_2d = separator.join(kombinasi_2d_list)
        tab2d, tab3d, tab4d = st.tabs([f"Kombinasi 2D ({len(kombinasi_2d_list)})", f"Kombinasi 3D ({len(kombinasi_3d_list)})", f"Kombinasi 4D ({len(kombinasi_4d_list)})"])
        with tab2d: st.text_area("Hasil 2D...", text_2d, height=200); st.download_button("Unduh 2D.txt", text_2d)
        with tab3d: st.text_area("Hasil 3D...", text_3d, height=200); st.download_button("Unduh 3D.txt", text_3d)
        with tab4d: st.text_area("Hasil 4D...", text_4d, height=200); st.download_button("Unduh 4D.txt", text_4d)
    
    # --- BAGIAN YANG DIUBAH ---
    st.subheader("🕵️ Analisis Tambahan")
    last_result_number = df['angka'].iloc[-1] if not df.empty else None
    new_controls_dict = calculate_new_controls(probs, last_result_number)
    if new_controls_dict:
        for label, numbers in new_controls_dict.items():
            if isinstance(numbers, list):
                # Ubah list of strings (untuk shio) atau list of ints menjadi string
                numbers_str = " ".join(map(str, numbers))
            else:
                # Untuk Jarak Lemah yang sudah string
                numbers_str = numbers
            st.markdown(f"#### **{label}:** `{numbers_str}`")
    st.divider()
    # --- AKHIR BAGIAN YANG DIUBAH ---
    
    st.subheader("💣 Rekomendasi Pola Permainan")
    # Menggunakan Angka Kontrol dari fungsi baru sebagai basis BBFS
    bbfs_digits = new_controls_dict.get("Angka Kontrol", [])
    if bbfs_digits:
        st.markdown(f"##### **BBFS 7 Digit (Rekomendasi):** `{' '.join(map(str, bbfs_digits))}`")
        try:
            # Generate 2D berdasarkan BBFS
            angka_jadi_2d_list = generate_angka_jadi_2d(probs, bbfs_digits)
            st.text_area(f"Top 2D Berdasarkan BBFS & Probabilitas...", " * ".join(angka_jadi_2d_list) if angka_jadi_2d_list else "-")
            
            # Generate 4D berdasarkan BBFS
            angka_jadi_4d_list = generate_angka_jadi_4d(probs, bbfs_digits)
            st.text_area(f"Top 4D Berdasarkan BBFS & Probabilitas...", " * ".join(angka_jadi_4d_list) if angka_jadi_4d_list else "-", height=200)

        except Exception as e: st.error(f"Galat saat generate angka jadi: {e}")
    st.divider()

# (Kode untuk Analisis Putaran Terbaik tetap sama, tidak perlu diubah)
if st.session_state.get('run_putaran_analysis', False):
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    # ... (sisa kode tidak berubah)
