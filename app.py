# app.py

import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pytz  # Library untuk timezone

# Impor fungsi yang sudah dimodifikasi untuk menerima parameter top_n
from markov_model import top_n_markov, top_n_markov_order2, top_n_markov_hybrid
from ai_model import (
    top_n_lstm,
    train_and_save_lstm,
    kombinasi_4d,
    top_n_ensemble,
    model_exists
)
from lokasi_list import lokasi_list

# Pastikan folder 'saved_models' ada
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

st.set_page_config(page_title="Prediksi Togel AI", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except requests.exceptions.RequestException:
        return None

# --- Bagian Judul dan Lottie ---
col_title, col_lottie = st.columns([4, 1])
with col_title:
    st.title("🔮 Prediksi 4D - AI & Markov")
with col_lottie:
    lottie_predict = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_g0nmp4p0.json")
    if lottie_predict:
        from streamlit_lottie import st_lottie
        st_lottie(lottie_predict, speed=1, height=100, key="prediksi")


# --- Sidebar Pengaturan ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"])
    putaran = st.number_input("🔁 Jumlah Putaran", min_value=10, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=10, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    
    # KODE BARU: Menambahkan input jumlah digit prediksi sesuai screenshot
    top_n_digits = st.number_input("📈 Jumlah Top Digit Prediksi", min_value=3, max_value=10, value=7)
    
    st.markdown("---")
    if st.button("🔍 Cari Putaran Terbaik"):
        # Placeholder untuk fungsionalitas di masa depan
        st.info("Fitur ini sedang dalam pengembangan!")

# --- Ambil Data ---
angka_list = []
if selected_lokasi and selected_hari:
    try:
        with st.spinner(f"🔄 Mengambil {putaran} data untuk pasaran {selected_lokasi}..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get("data"):
                angka_list = [item["result"] for item in data["data"] if len(item["result"]) == 4 and item["result"].isdigit()]
                st.success(f"✅ {len(angka_list)} data angka berhasil diambil untuk pasaran **{selected_lokasi}**.")
            else:
                st.warning("⚠️ Tidak ada data yang diterima dari API untuk konfigurasi ini.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Gagal mengambil data dari API: {e}")

df = pd.DataFrame({"angka": angka_list})

# ... (Sisa kode untuk Manajemen Model LSTM dan Tombol Prediksi disederhanakan di bawah)

# Tombol Prediksi
if st.button("🔮 Prediksi Sekarang!"):
    if len(df) < 20:
        st.warning("❌ Minimal 20 data diperlukan untuk melakukan prediksi.")
    else:
        with st.spinner("⏳ Mesin sedang menghitung prediksi..."):
            result = None
            # PERUBAHAN: Mengirimkan `top_n_digits` ke semua fungsi prediksi
            if metode == "Markov":
                result, _ = top_n_markov(df, top_n=top_n_digits)
            elif metode == "Markov Order-2":
                result = top_n_markov_order2(df, top_n=top_n_digits)
            elif metode == "Markov Gabungan":
                result = top_n_markov_hybrid(df, top_n=top_n_digits)
            elif metode == "LSTM AI":
                result = top_n_lstm(df, lokasi=selected_lokasi, top_n=top_n_digits)
            elif metode == "Ensemble AI + Markov":
                result = top_n_ensemble(df, lokasi=selected_lokasi, top_n=top_n_digits)

        if result is None:
            st.error("❌ Gagal melakukan prediksi. Pastikan model AI sudah dilatih jika menggunakan metode LSTM/Ensemble.")
        else:
            st.subheader(f"🎯 Hasil Prediksi Top {top_n_digits} Digit")
            col1, col2 = st.columns(2)
            digit_labels = ["Ribuan", "Ratusan", "Puluhan", "Satuan"]
            for i, label in enumerate(digit_labels):
                with (col1 if i < 2 else col2):
                    st.markdown(f"**{label}:**")
                    st.markdown(f"### {', '.join(map(str, result[i]))}")
            # ... (Sisa kode untuk kombinasi 4D dan evaluasi akurasi)


# --- KODE BARU: Footer untuk Lokasi & Waktu Server ---
st.markdown("---")
# Dapatkan timezone WIB
tz_wib = pytz.timezone('Asia/Jakarta')
now_wib = datetime.now(tz_wib)

# Lokalisasi manual ke Bahasa Indonesia
hari_id = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
bulan_id = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

nama_hari = hari_id[now_wib.weekday()]
nama_bulan = bulan_id[now_wib.month - 1]

# Format string waktu
waktu_server_str = f"{nama_hari}, {now_wib.day} {nama_bulan} {now_wib.year} {now_wib.strftime('%H:%M:%S')} WIB"
lokasi_server_str = "Brebes, Central Java, Indonesia"

# Tampilkan di aplikasi menggunakan markdown dengan sedikit style
st.markdown(
    f"""
    <div style="text-align: center; color: grey; font-size: 12px;">
        <p>📍 Lokasi & Waktu Server: {lokasi_server_str} - {waktu_server_str}</p>
    </div>
    """,
    unsafe_allow_html=True
)
