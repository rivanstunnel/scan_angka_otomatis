# app.py

import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from itertools import product

# ==== PERBAIKAN: Impor fungsi dengan nama yang sudah disinkronkan ====
from markov_model import (
    predict_markov,
    predict_markov_order2,
    predict_markov_hybrid,
)
from ai_model import (
    predict_lstm,
    train_and_save_lstm,
    kombinasi_4d,
    predict_ensemble,
    model_exists  # Pastikan model_exists ada di ai_model.py
)
from lokasi_list import lokasi_list
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
if lottie_predict:
    st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    putaran = st.number_input("🔁 Jumlah Putaran", min_value=1, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit Prediksi", min_value=1, max_value=9, value=6)

    min_conf = 0.0005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        st.subheader("Pengaturan Lanjutan (AI)")
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

    st.divider()
    if st.button("🔍 Latih Model AI"):
        with st.spinner(f"Melatih model AI untuk {selected_lokasi}, ini bisa memakan waktu beberapa menit..."):
            try:
                df_train = st.session_state.df_data
                train_and_save_lstm(df_train, selected_lokasi)
                st.success("✅ Model AI berhasil dilatih dan disimpan!")
            except Exception as e:
                st.error(f"❌ Gagal melatih model: {e}")

# --- (Sisa kode app.py sama, hanya pemanggilan fungsi prediksi yang diubah) ---
# Saya akan menyertakan bagian yang diubah saja untuk singkatnya.
# Salin semua kode di atas, dan ganti blok tombol prediksi dengan ini:

# --- Ambil Data ---
query_id = f"{selected_lokasi}-{selected_hari}"
if 'df_data' not in st.session_state or st.session_state.get('last_query') != query_id:
    with st.spinner("🔄 Mengambil data dari API..."):
        try:
            # URL dan headers Anda
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"} # Gantilah jika perlu
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            st.session_state.df_data = pd.DataFrame({"angka": all_angka})
            st.session_state.last_query = query_id
            st.success(f"✅ Total {len(st.session_state.df_data)} data berhasil diambil.")
        except Exception as e:
            st.error(f"❌ Gagal ambil data API: {e}")
            st.session_state.df_data = pd.DataFrame({"angka": []})

df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)

# Tombol Prediksi Utama
if st.button("🔮 Prediksi Sekarang!", use_container_width=True):
    # ==== PERBAIKAN: Pemanggilan fungsi disesuaikan dengan nama yang benar ====
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result = None
            if metode == "Markov": result, _ = predict_markov(df, top_n=top_n)
            elif metode == "Markov Order-2": result = predict_markov_order2(df, top_n=top_n)
            elif metode == "Markov Gabungan": result = predict_markov_hybrid(df, top_n=top_n)
            elif metode == "LSTM AI": result = predict_lstm(df, lokasi=selected_lokasi, top_n=top_n)
            elif metode == "Ensemble AI + Markov": result = predict_ensemble(df, lokasi=selected_lokasi, top_n=top_n)

        # ... (Sisa kode setelah ini bisa dibiarkan sama) ...
        # Pastikan Anda menyalin seluruh kode app.py yang saya berikan di atas
        # dan tempelkan ke file Anda
        if result is None:
            st.error("❌ Gagal melakukan prediksi. Untuk metode AI, pastikan model sudah dilatih terlebih dahulu.")
        else:
            st.subheader(f"🎯 Hasil Prediksi Top {top_n} Digit")
            # ... sisa kode visualisasi
            labels = ["As", "Kop", "Kepala", "Ekor"]
            for i, label in enumerate(labels):
                st.markdown(f"#### **{label}:** {', '.join(map(str, result[i]))}")

# ... (Pastikan Anda menyalin seluruh kode app.py yang saya berikan di atas) ...
