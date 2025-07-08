import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Impor fungsi-fungsi yang sudah dimodifikasi dari file model
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

# --- Sidebar ---
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    putaran = st.number_input("🔁 Jumlah Putaran", min_value=1, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)

    # Widget untuk memilih jumlah Top N Digit
    top_n = st.number_input("🔢 Jumlah Top Digit Prediksi", min_value=1, max_value=9, value=6)

    min_conf = 0.005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        st.subheader("Pengaturan Lanjutan (AI)")
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

# --- Ambil Data ---
angka_list = []
riwayat_input = ""
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"} # Ganti dengan Bearer Token Anda
            response = requests.get(url, headers=headers)
            data = response.json()
            angka_list = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            riwayat_input = "\n".join(angka_list)
            st.success(f"✅ {len(angka_list)} angka berhasil diambil.")
            with st.expander("📥 Lihat Data"):
                st.code(riwayat_input, language="text")
    except Exception as e:
        st.error(f"❌ Gagal ambil data API: {e}")

df = pd.DataFrame({"angka": angka_list})

# --- Manajemen Model LSTM ---
if metode in ["LSTM AI", "Ensemble AI + Markov"]:
    with st.expander("⚙️ Manajemen Model LSTM"):
        model_dir = "saved_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for i in range(4):
            model_path = os.path.join(model_dir, f"{selected_lokasi.lower().replace(' ', '_')}_digit{i}.h5")
            col1, col2 = st.columns([2, 1])
            with col1:
                if os.path.exists(model_path):
                    st.info(f"📂 Model Digit-{i+1} tersedia.")
                else:
                    st.warning(f"⚠️ Model Digit-{i+1} belum tersedia.")
            with col2:
                if os.path.exists(model_path):
                    if st.button(f"🗑 Hapus Model Digit-{i+1}", key=f"hapus_digit_{i}"):
                        os.remove(model_path)
                        st.rerun()

        if st.button("📚 Latih & Simpan Semua Model"):
            if len(df) < 20:
                 st.error("❌ Butuh minimal 20 data untuk melatih model.")
            else:
                with st.spinner("🔄 Melatih semua model per digit... Ini mungkin butuh waktu lama."):
                    train_and_save_lstm(df, selected_lokasi)
                st.success("✅ Semua model berhasil dilatih dan disimpan.")
                st.rerun()


# --- Tombol Prediksi ---
if st.button("🔮 Prediksi Sekarang!"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan untuk prediksi.")
    else:
        with st.spinner("⏳ Melakukan prediksi..."):
            result = None
            if metode == "Markov":
                result, _ = predict_markov(df, top_n=top_n)
            elif metode == "Markov Order-2":
                result = predict_markov_order2(df, top_n=top_n)
            elif metode == "Markov Gabungan":
                result = predict_markov_hybrid(df, top_n=top_n)
            elif metode == "LSTM AI":
                result = predict_lstm(df, lokasi=selected_lokasi, top_n=top_n)
            elif metode == "Ensemble AI + Markov":
                result = predict_ensemble(df, lokasi=selected_lokasi, top_n=top_n)

        if result is None:
            st.error("❌ Gagal melakukan prediksi. Pastikan model AI sudah dilatih jika menggunakan metode AI.")
        else:
            st.subheader(f"🎯 Hasil Prediksi Top {top_n} Digit")
            
            # --- MODIFIKASI: Mengubah label dan layout menjadi per baris ---
            labels = ["As", "Kop", "Kepala", "Ekor"]
            for i, label in enumerate(labels):
                st.markdown(f"#### **{label}:** {', '.join(map(str, result[i]))}")
            # --- AKHIR MODIFIKASI ---

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Menghitung kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, top_n=10, min_conf=min_conf, power=power)
                    if top_komb:
                        st.subheader("💡 Rekomendasi Kombinasi 4D")
                        sim_col = st.columns(2)
                        for i, (komb, score) in enumerate(top_komb):
                            with sim_col[i % 2]:
                                st.markdown(f"### `{komb}`\n*Confidence: `{score:.4f}`*")

        # --- Evaluasi Akurasi ---
        st.subheader("🔍 Evaluasi Akurasi Model")
        with st.spinner("📏 Menghitung akurasi berdasarkan data historis..."):
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total, benar = 0, 0
            akurasi_list = []
            
            # --- MODIFIKASI: Mengubah label di dictionary akurasi ---
            labels = ["As", "Kop", "Kepala", "Ekor"]
            digit_acc = {label: [] for label in labels}
            # --- AKHIR MODIFIKASI ---

            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df) - i)]
                if len(subset_df) < 11:
                    continue
                try:
                    pred = None
                    if metode == "Markov":
                        pred, _ = predict_markov(subset_df, top_n=top_n)
                    elif metode == "Markov Order-2":
                        pred = predict_markov_order2(subset_df, top_n=top_n)
                    elif metode == "Markov Gabungan":
                        pred = predict_markov_hybrid(subset_df, top_n=top_n)
                    elif metode == "LSTM AI":
                        pred = predict_lstm(subset_df, lokasi=selected_lokasi, top_n=top_n)
                    elif metode == "Ensemble AI + Markov":
                        pred = predict_ensemble(subset_df, lokasi=selected_lokasi, top_n=top_n)
                    
                    if pred is None: continue

                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    skor = 0
                    # --- MODIFIKASI: Menggunakan label baru di loop akurasi ---
                    for j, label in enumerate(labels):
                    # --- AKHIR MODIFIKASI ---
                        if int(actual[j]) in pred[j]:
                            skor += 1
                            digit_acc[label].append(1)
                        else:
                            digit_acc[label].append(0)
                    total += 4
                    benar += skor
                    akurasi_list.append(skor / 4 * 100)
                except Exception:
                    continue

            if total > 0:
                st.success(f"**📈 Akurasi Rata-rata ({metode})**: `{benar / total * 100:.2f}%`")
                
                tab1, tab2 = st.tabs(["Grafik Tren Akurasi", "Heatmap Akurasi per Digit"])
                with tab1:
                    st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                with tab2:
                    heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
                    fig, ax = plt.subplots()
                    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar=False)
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ Tidak cukup data historis untuk melakukan evaluasi akurasi.")
