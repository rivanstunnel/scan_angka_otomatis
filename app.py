# app.py

import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from itertools import product, permutations
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
def calculate_angka_kontrol(probabilities):
    """
    Menghitung Angka Kontrol berdasarkan matriks probabilitas.
    """
    if probabilities is None or probabilities.shape != (4, 10):
        return {}

    total_probs = np.sum(probabilities, axis=0)
    # --- PENAMBAHAN TOP 3D ---
    probs_3d = np.sum(probabilities[1:], axis=0) # Menjumlahkan probabilitas KOP, KEPALA, EKOR
    probs_2d = np.sum(probabilities[2:], axis=0)

    # 1. Angka Kontrol (AK) -> 7 digit
    ak_global = np.argsort(total_probs)[-7:][::-1].tolist()

    # --- PENAMBAHAN TOP 3D ---
    # Kalkulasi Top 3D
    top_3d = np.argsort(probs_3d)[-7:][::-1].tolist()

    # 2. Top 2D (KEP-EKO) -> 7 digit
    top_2d = np.argsort(probs_2d)[-7:][::-1].tolist()

    # 3. Top 4D (AS-KOP-KEP-EKO) -> 7 digit unik
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

    # 4. Angka Lemah (Hindari) -> 2 digit
    lemah_global = np.argsort(total_probs)[:2].tolist()

    # Mengembalikan kamus dengan urutan yang benar
    return {
        "Angka Kontrol (AK)": ak_global,
        "Top 4D (AS-KOP-KEP-EKO)": jagoan_final,
        # --- PENAMBAHAN TOP 3D ---
        "Top 3D (KOP-KEP-EKO)": top_3d,
        "Top 2D (KEP-EKO)": top_2d,
        "Angka Lemah (Hindari)": lemah_global,
    }

def generate_angka_jadi_2d(probabilities, bbfs_digits):
    """Menghasilkan semua kombinasi 2D dari digit BBFS dan mengurutkannya."""
    if probabilities is None or not bbfs_digits:
        return []
    all_2d_lines = list(product(bbfs_digits, repeat=2))
    scored_lines = []
    for line in all_2d_lines:
        kepala, ekor = int(line[0]), int(line[1])
        score = probabilities[2][kepala] + probabilities[3][ekor]
        scored_lines.append(("".join(map(str, line)), score))
    
    sorted_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)
    return [line[0] for line in sorted_lines]

def generate_angka_jadi_4d(probabilities, bbfs_source_digits):
    """Menghasilkan semua kombinasi 4D dari digit BBFS dan mengurutkannya."""
    if probabilities is None or not bbfs_source_digits:
        return []
    all_4d_lines = list(product(bbfs_source_digits, repeat=4))
    scored_lines = []
    for line in all_4d_lines:
        a, k, p, e = map(int, line)
        score = probabilities[0][a] + probabilities[1][k] + probabilities[2][p] + probabilities[3][e]
        scored_lines.append(("".join(map(str, line)), score))
        
    sorted_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)
    return [line[0] for line in sorted_lines]

# ==============================================================================
# --- UI (Tampilan Aplikasi) Dimulai di Sini ---
# ==============================================================================

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
        if total_data_saat_ini < jumlah_uji + 11:
            st.warning(f"Butuh minimal {jumlah_uji + 11} data riwayat untuk pengujian ini. Saat ini hanya ada **{total_data_saat_ini}** data.")
        else:
            st.session_state.run_putaran_analysis = True
            st.session_state.prediction_data = None

# ==============================================================================
# --- TAMPILAN UTAMA ---
# ==============================================================================

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

    with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
        kombinasi_4d_list = ["".join(map(str, p)) for p in product(*result)]
        kombinasi_3d_list = ["".join(map(str, p)) for p in product(*result[1:])]
        kombinasi_2d_list = ["".join(map(str, p)) for p in product(*result[2:])]
        separator = " * "
        text_4d = separator.join(kombinasi_4d_list)
        text_3d = separator.join(kombinasi_3d_list)
        text_2d = separator.join(kombinasi_2d_list)
        tab2d, tab3d, tab4d = st.tabs([f"Kombinasi 2D ({len(kombinasi_2d_list)})", f"Kombinasi 3D ({len(kombinasi_3d_list)})", f"Kombinasi 4D ({len(kombinasi_4d_list)})"])

        with tab2d:
            st.text_area("Hasil 2D (Kepala-Ekor)", text_2d, height=200)
            st.download_button("Unduh 2D.txt", text_2d, file_name="hasil_2d.txt")
        with tab3d:
            st.text_area("Hasil 3D (Kop-Kepala-Ekor)", text_3d, height=200)
            st.download_button("Unduh 3D.txt", text_3d, file_name="hasil_3d.txt")
        with tab4d:
            st.text_area("Hasil 4D (As-Kop-Kepala-Ekor)", text_4d, height=200)
            st.download_button("Unduh 4D.txt", text_4d, file_name="hasil_4d.txt")
    
    angka_kontrol_dict = calculate_angka_kontrol(probs)
    if angka_kontrol_dict:
        st.subheader("🕵️ Angka Kontrol")
        # Loop ini akan secara otomatis menampilkan Top 3D karena sudah ada di dalam dict
        for label, numbers in angka_kontrol_dict.items():
            numbers_str = " ".join(map(str, numbers))
            st.markdown(f"#### **{label}:** `{numbers_str}`")
        st.divider()

        st.subheader("💣 Rekomendasi Pola Permainan")
        
        # --- Bagian 2D ---
        bbfs_digits_2d = angka_kontrol_dict.get("Top 2D (KEP-EKO)", [])[:7]
        if bbfs_digits_2d:
            bbfs_str_2d = " ".join(map(str, bbfs_digits_2d))
            st.markdown(f"##### **BBFS 7 Digit (2D):** `{bbfs_str_2d}`")
            try:
                angka_jadi_2d_list = generate_angka_jadi_2d(probs, bbfs_digits_2d)
                angka_jadi_2d_str = " * ".join(angka_jadi_2d_list) if angka_jadi_2d_list else "-"
                st.text_area(f"Angka Jadi 2D ({len(angka_jadi_2d_list)} Line)", value=angka_jadi_2d_str)
            except Exception as e:
                st.error(f"Terjadi galat saat membuat Angka Jadi 2D: {e}")
        
        # --- Bagian 4D ---
        bbfs_digits_4d = angka_kontrol_dict.get("Top 4D (AS-KOP-KEP-EKO)", [])[:7]
        if bbfs_digits_4d:
            bbfs_str_4d = " ".join(map(str, bbfs_digits_4d))
            st.markdown(f"##### **BBFS 7 Digit (4D):** `{bbfs_str_4d}`")
            try:
                angka_jadi_4d_list = generate_angka_jadi_4d(probs, bbfs_digits_4d)
                angka_jadi_4d_str = " * ".join(angka_jadi_4d_list) if angka_jadi_4d_list else "-"
                st.text_area(f"Angka Jadi 4D ({len(angka_jadi_4d_list)} Line)", value=angka_jadi_4d_str, height=200)
            except Exception as e:
                st.error(f"Terjadi galat saat membuat Angka Jadi 4D: {e}")
        st.divider()

# --- Bagian Analisis Putaran Terbaik ---
# Sisa kode ini tidak diubah dan tidak perlu disalin ulang jika sudah benar
if st.session_state.get('run_putaran_analysis', False):
    # ... (kode analisis putaran yang sudah ada) ...
    pass
