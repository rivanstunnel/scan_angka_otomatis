# app.py

import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from itertools import product, combinations, permutations

# Impor fungsi-fungsi dari file model.
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
def calculate_angka_kontrol(probabilities, n=6):
    """
    Menghitung 4 baris Angka Kontrol yang berbeda berdasarkan matriks probabilitas.
    """
    if probabilities is None or probabilities.shape != (4, 10):
        return {}
    total_probs = np.sum(probabilities, axis=0)
    ak_global = np.argsort(total_probs)[-n:][::-1].tolist()
    probs_2d = np.sum(probabilities[2:], axis=0)
    top_2d = np.argsort(probs_2d)[-n:][::-1].tolist()
    kuat_posisi = np.argmax(probabilities, axis=1).tolist()
    lemah_global = np.argsort(total_probs)[:n].tolist()
    return {
        "Angka Kontrol (AK)": ak_global,
        "Top 2D (KEP-EKO)": top_2d,
        "Jagoan Posisi (AS-KOP-KEP-EKO)": kuat_posisi,
        "Angka Lemah (Hindari)": lemah_global,
    }

def generate_angka_jadi_2d(probabilities, bbfs_digits, n_lines=10):
    """Menghasilkan 10 line Angka Jadi 2D dari digit BBFS."""
    if probabilities is None or not bbfs_digits:
        return []
    all_2d_lines = list(permutations(bbfs_digits, 2))
    scored_lines = []
    for line in all_2d_lines:
        kepala, ekor = line
        score = probabilities[2][kepala] + probabilities[3][ekor]
        scored_lines.append(("".join(map(str, line)), score))
    sorted_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)
    return [line[0] for line in sorted_lines[:n_lines]]

def generate_bom_4d(probabilities, result_digits, n_lines=10):
    """Menghasilkan 10 line Bom 4D dari hasil analisis."""
    if probabilities is None or not result_digits or len(result_digits) != 4:
        return []
    all_4d_lines = list(product(*result_digits))
    scored_lines = []
    for line in all_4d_lines:
        a, k, p, e = line
        score = probabilities[0][a] + probabilities[1][k] + probabilities[2][p] + probabilities[3][e]
        scored_lines.append(("".join(map(str, line)), score))
    sorted_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)
    return [line[0] for line in sorted_lines[:n_lines]]

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
        if total_data_saat_ini < 30:
            st.warning(f"Butuh minimal 30 data riwayat. Saat ini hanya ada **{total_data_saat_ini}** data yang dimuat.")
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
    
    angka_kontrol_dict = calculate_angka_kontrol(probs, n=6)
    if angka_kontrol_dict:
        st.subheader("🕵️ Angka Kontrol")
        for label, numbers in angka_kontrol_dict.items():
            numbers_str = " ".join(map(str, numbers))
            st.markdown(f"#### **{label}:** `{numbers_str}`")
        st.divider()

        # --- BAGIAN REKOMENDASI PERMAINAN (FORMAT DIUBAH) ---
        st.subheader("💣 Rekomendasi Pola Permainan")

        # 1. BBFS 5 Digit (dari Top 2D)
        bbfs_digits = angka_kontrol_dict.get("Top 2D (KEP-EKO)", [])[:5]
        if bbfs_digits:
            bbfs_str = " ".join(map(str, bbfs_digits))
            st.markdown(f"##### **BBFS 5 Digit (2D):** `{bbfs_str}`")
        
        # 2. Angka Jadi 2D (10 Line)
        angka_jadi_2d = generate_angka_jadi_2d(probs, bbfs_digits, n_lines=10)
        if angka_jadi_2d:
            st.text_area(
                "Angka Jadi 2D (10 Line)",
                " * ".join(angka_jadi_2d), # Pemisah diubah menjadi bintang
                height=40,
                help="10 set 2D terbaik berdasarkan BBFS, dipisahkan oleh bintang."
            )
        
        # 3. Bom 4D (10 Line)
        bom_4d_lines = generate_bom_4d(probs, result, n_lines=10)
        if bom_4d_lines:
            st.text_area(
                "Bom 4D (10 Line)",
                " * ".join(bom_4d_lines), # Pemisah diubah menjadi bintang
                height=40,
                help="10 set 4D terbaik berdasarkan probabilitas, dipisahkan oleh bintang."
            )

        st.divider()
        # --- AKHIR BAGIAN REKOMENDASI ---

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

if st.session_state.get('run_putaran_analysis', False):
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    with st.spinner("Menganalisis berbagai jumlah putaran..."):
        full_df = st.session_state.get('df_data', pd.DataFrame())
        putaran_results = {}
        max_putaran_test = len(full_df) - jumlah_uji
        
        if max_putaran_test < 20:
             st.warning(f"Data tidak cukup. Butuh total {20 + jumlah_uji} data, hanya ada {len(full_df)}.")
        else:
            test_range = list(range(20, max_putaran_test + 1, 10))
            if max_putaran_test not in test_range and max_putaran_test > 20:
                test_range.append(max_putaran_test)

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
