# Simpan sebagai app.py

import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Impor fungsi yang sudah dimodifikasi menjadi top7
from markov_model import (
    top7_markov, 
    top7_markov_order2, 
    top7_markov_hybrid
)
from ai_model import (
    top7_lstm,
    train_and_save_lstm,
    kombinasi_4d,
    top7_ensemble,
    model_exists
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
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

# Sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    putaran = st.number_input("🔁 Jumlah Putaran", min_value=1, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

    min_conf = 0.005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

# Ambil Data
angka_list = []
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code("\n".join(angka_list), language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

df = pd.DataFrame({"angka": angka_list})

# Manajemen Model LSTM
if metode in ["LSTM AI", "Ensemble AI + Markov"]:
    with st.expander("⚙️ Manajemen Model LSTM"):
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
            st.info("Folder 'saved_models' dibuat.")

        for i in range(4):
            model_path = f"saved_models/{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5"
            col1, col2 = st.columns([2, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"📂 Model Digit-{i} tersedia.")
                else:
                    st.warning(f"⚠️ Model Digit-{i} belum tersedia.")
            with col2:
                if os.path.exists(model_path):
                    if st.button(f"🗑 Hapus Digit-{i}", key=f"hapus_digit_{i}"):
                        os.remove(model_path)
                        st.experimental_rerun()

        if st.button("📚 Latih & Simpan Semua Model"):
            if len(df) > 20:
                with st.spinner("🔄 Melatih semua model per digit..."):
                    train_and_save_lstm(df, selected_lokasi)
                st.success("✅ Semua model berhasil dilatih dan disimpan.")
                st.experimental_rerun()
            else:
                st.warning("⚠️ Minimal 21 data diperlukan untuk melatih model.")


# Tombol Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result = None
            if metode == "Markov":
                result, _ = top7_markov(df)
            elif metode == "Markov Order-2":
                result = top7_markov_order2(df)
            elif metode == "Markov Gabungan":
                result = top7_markov_hybrid(df)
            elif metode == "LSTM AI":
                if model_exists(selected_lokasi):
                    result = top7_lstm(df, lokasi=selected_lokasi)
                else:
                    st.error("❌ Model LSTM belum dilatih. Silakan latih model terlebih dahulu di 'Manajemen Model LSTM'.")
            elif metode == "Ensemble AI + Markov":
                if model_exists(selected_lokasi):
                    result = top7_ensemble(df, lokasi=selected_lokasi)
                else:
                    st.error("❌ Model LSTM untuk Ensemble belum dilatih. Silakan latih model terlebih dahulu.")

        if result is None:
            if metode not in ["LSTM AI", "Ensemble AI + Markov"]:
                 st.error("❌ Gagal melakukan prediksi.")
        else:
            # MODIFIKASI UTAMA: Judul diubah menjadi Top 7
            with st.expander("🎯 Hasil Prediksi Top 7 Digit", expanded=True):
                col1, col2 = st.columns(2)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with (col1 if i % 2 == 0 else col2):
                        st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    if top_komb:
                        with st.expander("💡 Simulasi Kombinasi 4D Terbaik"):
                            sim_col = st.columns(2)
                            for i, (komb, score) in enumerate(top_komb):
                                with sim_col[i % 2]:
                                    st.markdown(f"`{komb}` - ⚡️ Confidence: `{score:.4f}`")

            # Evaluasi Akurasi
            with st.spinner("📏 Menghitung akurasi..."):
                uji_df = df.tail(min(jumlah_uji, len(df)-1))
                if len(uji_df) > 0:
                    total, benar = 0, 0
                    akurasi_list = []
                    digit_acc = {"Ribuan": [], "Ratusan": [], "Puluhan": [], "Satuan": []}

                    for i in range(len(uji_df)):
                        subset_df = df.iloc[:-(len(uji_df) - i)]
                        if len(subset_df) < 20: continue
                        
                        try:
                            pred = None
                            if metode == "Markov": pred, _ = top7_markov(subset_df)
                            elif metode == "Markov Order-2": pred = top7_markov_order2(subset_df)
                            elif metode == "Markov Gabungan": pred = top7_markov_hybrid(subset_df)
                            elif metode == "LSTM AI": pred = top7_lstm(subset_df, lokasi=selected_lokasi)
                            elif metode == "Ensemble AI + Markov": pred = top7_ensemble(subset_df, lokasi=selected_lokasi)
                            
                            if pred is None: continue

                            actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                            skor = 0
                            for j, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                                if int(actual[j]) in pred[j]:
                                    skor += 1
                                    digit_acc[label].append(1)
                                else:
                                    digit_acc[label].append(0)
                            total += 4
                            benar += skor
                            akurasi_list.append(skor / 4 * 100)
                        except Exception as e:
                            # st.warning(f"Error saat evaluasi baris {i}: {e}")
                            continue

                    if total > 0:
                        st.success(f"📈 Akurasi {metode} (dari {len(uji_df)} data uji): {benar / total * 100:.2f}%")
                        with st.expander("📊 Grafik Akurasi"):
                            st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                        with st.expander("🔥 Heatmap Akurasi per Digit"):
                            heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
                            fig, ax = plt.subplots()
                            sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
                            st.pyplot(fig)
                    else:
                        st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi.")
