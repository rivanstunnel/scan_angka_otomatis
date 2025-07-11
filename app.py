# app.py

import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
from itertools import product

# Impor fungsi-fungsi dari file model.
from markov_model import (
    predict_markov,
    predict_markov_order2,
    predict_markov_hybrid,
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Prediksi 4D Markov", layout="wide")

# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException:
        return None

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
if lottie_predict:
    st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - Metode Markov")

# --- Daftar Opsi ---
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan"]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # ==== FITUR BARU: Pilihan Sumber Data ====
    data_source = st.radio(
        " sumber Data",
        ("API", "Input Manual"),
        horizontal=True,
    )
    
    # Kontrol yang muncul berdasarkan pilihan sumber data
    if data_source == "API":
        hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
        selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
        selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    else: # data_source == "Input Manual"
        manual_data_input = st.text_area(
            "📋 Masukkan Data Keluaran",
            height=150,
            placeholder="Contoh: 1234 5678, 9012\nPisahkan angka dengan spasi, koma, atau baris baru."
        )

    st.divider()
    putaran = st.number_input("🔁 Jumlah Data Terakhir Digunakan", min_value=1, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit Prediksi", min_value=1, max_value=9, value=6)

# --- Logika Pengambilan dan Pemrosesan Data ---
# Dijalankan berdasarkan pilihan di sidebar

# 1. Logika untuk Sumber Data API
if data_source == "API":
    query_id = f"{selected_lokasi}-{selected_hari}"
    if 'df_data' not in st.session_state or st.session_state.get('last_query') != query_id:
        with st.spinner(f"🔄 Mengambil data untuk pasaran {selected_lokasi}..."):
            try:
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                data = response.json()
                all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
                st.session_state.df_data = pd.DataFrame({"angka": all_angka})
                st.session_state.last_query = query_id
                st.success(f"✅ Total {len(st.session_state.df_data)} data berhasil diambil dari API.")
            except Exception as e:
                st.error(f"❌ Gagal ambil data API: {e}")
                st.session_state.df_data = pd.DataFrame({"angka": []})

# 2. Logika untuk Sumber Data Manual
elif data_source == "Manual":
    # Hanya proses ulang jika teks di dalam box berubah
    if st.session_state.get('last_manual_input') != manual_data_input:
        # Menggunakan regex untuk menemukan semua angka 4 digit
        angka_list = re.findall(r'\b\d{4}\b', manual_data_input)
        
        # Simpan ke session state
        st.session_state.df_data = pd.DataFrame({"angka": angka_list})
        st.session_state.last_manual_input = manual_data_input
        st.session_state.last_query = "manual" # Tandai bahwa sumbernya manual
        if angka_list:
            st.success(f"✅ Total {len(angka_list)} data berhasil diproses dari input manual.")

# Mengambil data dari session state untuk digunakan di seluruh aplikasi
df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)

# Tampilkan data yang digunakan jika ada
if not df.empty:
    with st.expander(f"✅ Menampilkan {len(df)} data terakhir yang digunakan."):
        st.code("\n".join(df['angka'].tolist()), language="text")

# --- Tombol Prediksi Utama ---
if st.button("🔮 Prediksi Sekarang!", use_container_width=True):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result = None
            if metode == "Markov": result, _ = predict_markov(df, top_n=top_n)
            elif metode == "Markov Order-2": result = predict_markov_order2(df, top_n=top_n)
            elif metode == "Markov Gabungan": result = predict_markov_hybrid(df, top_n=top_n)

        if result is None:
            st.error("❌ Gagal melakukan prediksi.")
        else:
            st.subheader(f"🎯 Hasil Prediksi Top {top_n} Digit")
            labels = ["As", "Kop", "Kepala", "Ekor"]
            cols = st.columns(4)
            for i, label in enumerate(labels):
                with cols[i]:
                    st.metric(label, ", ".join(map(str, result[i])))

            st.divider()
            with st.expander("⬇️ Tampilkan & Unduh Hasil Kombinasi"):
                # (Sisa kode untuk kombinasi dan evaluasi tidak berubah)
                kombinasi_4d_list = ["".join(map(str, p)) for p in product(*result)]
                kombinasi_3d_list = ["".join(map(str, p)) for p in product(*result[1:])]
                kombinasi_2d_list = ["".join(map(str, p)) for p in product(*result[2:])]

                separator = " * "
                text_4d = separator.join(kombinasi_4d_list)
                text_3d = separator.join(kombinasi_3d_list)
                text_2d = separator.join(kombinasi_2d_list)

                tab4d, tab3d, tab2d = st.tabs([f"Kombinasi 4D ({len(kombinasi_4d_list)})", f"Kombinasi 3D ({len(kombinasi_3d_list)})", f"Kombinasi 2D ({len(kombinasi_2d_list)})"])

                with tab4d:
                    st.text_area("Hasil 4D (As-Kop-Kepala-Ekor)", text_4d, height=200)
                    st.download_button("Unduh 4D.txt", text_4d, file_name="hasil_4d.txt")

                with tab3d:
                    st.text_area("Hasil 3D (Kop-Kepala-Ekor)", text_3d, height=200)
                    st.download_button("Unduh 3D.txt", text_3d, file_name="hasil_3d.txt")

                with tab2d:
                    st.text_area("Hasil 2D (Kepala-Ekor)", text_2d, height=200)
                    st.download_button("Unduh 2D.txt", text_2d, file_name="hasil_2d.txt")

            # Blok Evaluasi Akurasi
            st.subheader("🔍 Evaluasi Akurasi Model")
            with st.spinner("📏 Menghitung akurasi..."):
                uji_df = df.tail(min(jumlah_uji, len(df)))
                total_eval, benar_eval = 0, 0
                
                if len(uji_df) > 0:
                    for i in range(len(uji_df)):
                        subset_df = df.iloc[:-(len(uji_df) - i)]
                        if len(subset_df) < 11: continue
                        try:
                            pred_eval = None
                            if metode == "Markov": pred_eval, _ = predict_markov(subset_df, top_n=top_n)
                            elif metode == "Markov Order-2": pred_eval = predict_markov_order2(subset_df, top_n=top_n)
                            elif metode == "Markov Gabungan": pred_eval = predict_markov_hybrid(subset_df, top_n=top_n)

                            if pred_eval is None: continue
                            actual_eval = f"{int(uji_df.iloc[i]['angka']):04d}"
                            
                            for k in range(4):
                                if int(actual_eval[k]) in pred_eval[k]:
                                    benar_eval += 1
                            total_eval += 4
                        except Exception: continue

                if total_eval > 0:
                    st.success(f"**📈 Akurasi Rata-rata (Top-{top_n})**: `{benar_eval / total_eval * 100:.2f}%`")
                else:
                    st.warning("⚠️ Tidak cukup data untuk melakukan evaluasi akurasi.")
