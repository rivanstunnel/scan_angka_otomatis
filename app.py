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
# Pastikan Anda memiliki file lokasi_list.py di direktori yang sama
# Contoh isi file lokasi_list.py:
# lokasi_list = ["HONGKONG", "SYDNEY", "SINGAPORE"]
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
def calculate_angka_kontrol(probabilities, top_n=7):
    """
    PERBAIKAN: Menghitung Angka Kontrol berdasarkan matriks probabilitas.
    Jumlah digit yang dihasilkan kini dinamis sesuai dengan parameter top_n.
    """
    if probabilities is None or probabilities.shape != (4, 10):
        return {}

    total_probs = np.sum(probabilities, axis=0)
    probs_2d = np.sum(probabilities[2:], axis=0)

    # 1. Angka Kontrol (AK) -> Jumlah digit mengikuti top_n
    ak_global = np.argsort(total_probs)[-top_n:][::-1].tolist()

    # 2. Top 2D (KEP-EKO) -> Jumlah digit mengikuti top_n
    top_2d = np.argsort(probs_2d)[-top_n:][::-1].tolist()

    # 3. Jagoan Posisi (AS-KOP-KEP-EKO) -> Jumlah digit unik mengikuti top_n
    jagoan_per_posisi = np.argmax(probabilities, axis=1).tolist()
    jagoan_final = list(dict.fromkeys(jagoan_per_posisi))
    
    for digit in ak_global:
        if len(jagoan_final) >= top_n: break
        if digit not in jagoan_final: jagoan_final.append(digit)
            
    if len(jagoan_final) < top_n:
        sisa_digit = [d for d in range(10) if d not in jagoan_final]
        needed = top_n - len(jagoan_final)
        jagoan_final.extend(sisa_digit[:needed])

    # 4. Angka Lemah (Hindari) -> Dibiarkan 2 digit
    lemah_global = np.argsort(total_probs)[:2].tolist()

    return {
        "Angka Kontrol (AK)": ak_global,
        "Top 2D (KEP-EKO)": top_2d,
        "Jagoan Posisi (AS-KOP-KEP-EKO)": jagoan_final,
        "Angka Lemah (Hindari)": lemah_global,
    }

def generate_angka_jadi_2d(probabilities, bbfs_digits):
    """Menghasilkan semua kombinasi 2D dari digit BBFS dan mengurutkannya."""
    if probabilities is None or not bbfs_digits: return []
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
    if probabilities is None or not bbfs_source_digits: return []
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
    data_source = st.radio("Sumber Data", ("API", "Input Manual"), horizontal=True, key='data_source_selector')

    if data_source == "API":
        hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
        selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
        selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
        if st.button("Muat Data API"):
            with st.spinner(f"🔄 Mengambil data untuk {selected_lokasi}..."):
                try:
                    url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
                    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"} # Ganti dengan API Key Anda
                    response = requests.get(url, headers=headers, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                    st.session_state.df_data = pd.DataFrame({"angka": all_angka})
                    st.session_state.prediction_data = None
                    st.success(f"Berhasil memuat {len(all_angka)} data.")
                except Exception as e:
                    st.error(f"❌ Gagal ambil data API: {e}")
                    st.session_state.df_data = pd.DataFrame()
    else: # Input Manual
        manual_data_input = st.text_area("📋 Masukkan Data Keluaran (pisahkan dengan spasi, koma, atau baris baru)", height=150, placeholder="Contoh: 1234 5678, 9012...")
        if st.button("Proses Data Manual"):
            angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
            st.session_state.df_data = pd.DataFrame({"angka": angka_list})
            st.session_state.prediction_data = None
            st.success(f"Berhasil memproses {len(angka_list)} data.")

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", 1, 1000, 100)
    metode = st.selectbox("🧠 Metode Analisis", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit", 1, 9, 7)
    st.divider()

    st.header("🔬 Analisis Lanjutan")
    jumlah_uji = st.number_input("📊 Jml Data untuk Back-testing", 1, 200, 10, help="Berapa banyak data terakhir yang akan dijadikan 'kunci jawaban' untuk menguji akurasi.")
    if st.button("🔍 Analisis Putaran Terbaik"):
        total_data_saat_ini = len(st.session_state.get('df_data', []))
        if total_data_saat_ini < jumlah_uji + 11:
            st.warning(f"Butuh minimal {jumlah_uji + 11} data riwayat. Saat ini hanya ada {total_data_saat_ini} data.")
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
        with st.spinner("⏳ Menganalisis..."):
            result, probs = None, None
            if metode == "Markov": result, probs = predict_markov(df, top_n=top_n)
            elif metode == "Markov Order-2": result, probs = predict_markov_order2(df, top_n=top_n)
            elif metode == "Markov Gabungan": result, probs = predict_markov_hybrid(df, top_n=top_n)
            
            if result is not None:
                st.session_state.prediction_data = {"result": result, "probs": probs}
            else:
                st.error("Gagal melakukan analisis. Periksa kembali data input.")

if st.session_state.get('prediction_data') is not None:
    prediction_data = st.session_state.prediction_data
    result = prediction_data["result"]
    probs = prediction_data["probs"]

    st.subheader(f"🎯 Hasil Analisis Top {top_n} Digit")
    labels = ["As", "Kop", "Kepala", "Ekor"]
    cols = st.columns(4)
    for i, label in enumerate(labels):
        with cols[i]:
            hasil_str = ", ".join(map(str, result[i]))
            st.markdown(f"#### **{label}:**")
            st.code(hasil_str, language="text")
    st.divider()
    
    # PERBAIKAN: Pemanggilan fungsi dengan parameter top_n
    angka_kontrol_dict = calculate_angka_kontrol(probs, top_n)
    if angka_kontrol_dict:
        st.subheader("🕵️ Angka Kontrol & Rekomendasi Pola")
        
        col1, col2 = st.columns(2)
        with col1:
            for label, numbers in angka_kontrol_dict.items():
                numbers_str = " ".join(map(str, numbers))
                st.markdown(f"**{label}:** `{numbers_str}`")

        with col2:
            bbfs_digits_4d = angka_kontrol_dict.get("Jagoan Posisi (AS-KOP-KEP-EKO)", [])[:7]
            if bbfs_digits_4d:
                bbfs_str_4d = " ".join(map(str, bbfs_digits_4d))
                st.markdown(f"**BBFS 7 Digit (4D):** `{bbfs_str_4d}`")

            bbfs_digits_2d = angka_kontrol_dict.get("Top 2D (KEP-EKO)", [])[:7]
            if bbfs_digits_2d:
                bbfs_str_2d = " ".join(map(str, bbfs_digits_2d))
                st.markdown(f"**BBFS 7 Digit (2D):** `{bbfs_str_2d}`")
        st.divider()

    st.subheader("🎰 Angka Jadi & Kombinasi")
    tab2d, tab3d, tab4d = st.tabs(["Angka Jadi 2D", "Angka Jadi 4D", "Kombinasi Fullset"])
    
    with tab2d:
        bbfs_2d = angka_kontrol_dict.get("Top 2D (KEP-EKO)", [])[:7]
        if bbfs_2d:
            angka_jadi_2d_list = generate_angka_jadi_2d(probs, bbfs_2d)
            angka_jadi_2d_str = " ".join(angka_jadi_2d_list) if angka_jadi_2d_list else "-"
            st.text_area(f"Urutan 2D Terbaik dari BBFS 7 Digit ({len(angka_jadi_2d_list)} Line)", value=angka_jadi_2d_str, height=150)
        else:
            st.info("Angka Kontrol belum dihasilkan untuk membuat Angka Jadi.")

    with tab4d:
        kombinasi_4d_list = ["".join(map(str, p)) for p in product(*result)]
        st.text_area(f"Kombinasi 4D dari Top {top_n} Digit ({len(kombinasi_4d_list)} Line)", " ".join(kombinasi_4d_list), height=200)

    with tab3d:
        bbfs_4d = angka_kontrol_dict.get("Jagoan Posisi (AS-KOP-KEP-EKO)", [])[:7]
        if bbfs_4d:
            angka_jadi_4d_list = generate_angka_jadi_4d(probs, bbfs_4d)
            angka_jadi_4d_str = " ".join(angka_jadi_4d_list) if angka_jadi_4d_list else "-"
            st.text_area(f"Urutan 4D Terbaik dari BBFS 7 Digit ({len(angka_jadi_4d_list)} Line)", value=angka_jadi_4d_str, height=200)
        else:
            st.info("Angka Kontrol belum dihasilkan untuk membuat Angka Jadi.")
    st.divider()

# --- Bagian Analisis Putaran Terbaik ---
if st.session_state.get('run_putaran_analysis', False):
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    with st.spinner("Menganalisis berbagai skenario putaran... Ini mungkin akan memakan waktu."):
        full_df = st.session_state.get('df_data', pd.DataFrame())
        putaran_results = {}
        max_putaran_test = len(full_df) - jumlah_uji
        start_putaran = 11
        end_putaran = max_putaran_test
        
        if end_putaran < start_putaran:
            st.warning(f"Data tidak cukup. Butuh setidaknya {start_putaran + jumlah_uji} total data riwayat.")
        else:
            test_range = list(range(start_putaran, end_putaran + 1, 1))
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
                    # PERBAIKAN BUG: Gunakan `train_df_for_step` bukan `df`
                    if metode == "Markov": pred, _ = predict_markov(train_df_for_step, top_n=top_n)
                    elif metode == "Markov Order-2": pred, _ = predict_markov_order2(train_df_for_step, top_n=top_n)
                    elif metode == "Markov Gabungan": pred, _ = predict_markov_hybrid(train_df_for_step, top_n=top_n)

                    if pred:
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
                m2.metric("Akurasi Tertinggi", f"{best_accuracy:.2f}%", f"Berdasarkan pengujian {jumlah_uji} data terakhir")

                chart_data = pd.DataFrame.from_dict(putaran_results, orient='index', columns=['Akurasi (%)'])
                chart_data.index.name = 'Jumlah Putaran'
                st.line_chart(chart_data)

    st.session_state.run_putaran_analysis = False
