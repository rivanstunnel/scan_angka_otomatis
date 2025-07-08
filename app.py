import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product # <-- PERUBAHAN BARU: Impor library product

# Impor fungsi top7
from markov_model import top7_markov, top7_markov_order2, top7_markov_hybrid
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
riwayat_input = ""
if selected_lokasi and selected_hari:
    try:
        with st.spinner("🔄 Mengambil data dari API..."):
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
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

# Manajemen Model LSTM
if metode in ["LSTM AI", "Ensemble AI + Markov"]:
    with st.expander("⚙️ Manajemen Model LSTM"):
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
                        st.warning(f"✅ Model Digit-{i} dihapus.")
                        st.experimental_rerun()

        if st.button("📚 Latih & Simpan Semua Model"):
            with st.spinner("🔄 Melatih semua model per digit..."):
                train_and_save_lstm(df, selected_lokasi)
            st.success("✅ Semua model berhasil dilatih dan disimpan.")
            st.experimental_rerun()

# Tombol Prediksi
if st.button("🔮 Prediksi"):
    if len(df) < 11:
        st.warning("❌ Minimal 11 data diperlukan.")
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
                result = top7_lstm(df, lokasi=selected_lokasi)
            elif metode == "Ensemble AI + Markov":
                result = top7_ensemble(df, lokasi=selected_lokasi)

        if result is None:
            st.error("❌ Gagal melakukan prediksi. Pastikan model AI sudah dilatih jika menggunakan metode tersebut.")
        else:
            with st.expander("🎯 Hasil Prediksi Top 7 Digit", expanded=True):
                labels = ["As", "Kop", "Kepala", "Ekor"]
                for i, label in enumerate(labels):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            # --- PERUBAHAN BARU: Menambahkan expander untuk Angka Jadi 2D, 3D, 4D ---
            with st.expander("🔢 Angka Jadi 2D, 3D, 4D", expanded=True):
                # Ekstrak hasil prediksi untuk setiap posisi
                as_pred, kop_pred, kepala_pred, ekor_pred = result[0], result[1], result[2], result[3]

                # Generate Angka Jadi 2D (dari semua 7x7 prediksi Kepala & Ekor)
                kombinasi_2d = product(kepala_pred, ekor_pred)
                angka_jadi_2d = sorted(["".join(map(str, k)) for k in kombinasi_2d])
                st.markdown("**Angka Jadi 2D:**")
                st.code("*".join(angka_jadi_2d))

                # Generate Angka Jadi 3D (dari top 4 prediksi Kop, Kepala, Ekor)
                kombinasi_3d = product(kop_pred[:4], kepala_pred[:4], ekor_pred[:4])
                angka_jadi_3d = sorted(["".join(map(str, k)) for k in kombinasi_3d])
                st.markdown("**Angka Jadi 3D:**")
                st.code("*".join(angka_jadi_3d))
                
                # Generate Angka Jadi 4D (dari top 3 prediksi As, Kop, Kepala, Ekor)
                kombinasi_4d_jadi = product(as_pred[:3], kop_pred[:3], kepala_pred[:3], ekor_pred[:3])
                angka_jadi_4d = sorted(["".join(map(str, k)) for k in kombinasi_4d_jadi])
                st.markdown("**Angka Jadi 4D:**")
                st.code("*".join(angka_jadi_4d))
            # --- Akhir Perubahan Baru ---

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
            uji_df = df.tail(min(jumlah_uji, len(df)))
            total, benar = 0, 0
            akurasi_list = []
            
            labels_acc = ["As", "Kop", "Kepala", "Ekor"]
            digit_acc = {label: [] for label in labels_acc}

            for i in range(len(uji_df)):
                subset_df = df.iloc[:-(len(uji_df) - i)]
                if len(subset_df) < 20:
                    continue
                try:
                    pred = (
                        top7_markov(subset_df)[0] if metode == "Markov" else
                        top7_markov_order2(subset_df) if metode == "Markov Order-2" else
                        top7_markov_hybrid(subset_df) if metode == "Markov Gabungan" else
                        top7_lstm(subset_df, lokasi=selected_lokasi) if metode == "LSTM AI" else
                        top7_ensemble(subset_df, lokasi=selected_lokasi)
                    )
                    if pred is None:
                        continue

                    actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                    skor = 0
                    for j, label in enumerate(labels_acc):
                        if int(actual[j]) in pred[j]:
                            skor += 1
                            digit_acc[label].append(1)
                        else:
                            digit_acc[label].append(0)
                    total += 4
                    benar += skor
                    akurasi_list.append(skor / 4 * 100)
                except:
                    continue

            if total > 0:
                st.success(f"📈 Akurasi {metode}: {benar / total * 100:.2f}%")
                with st.expander("📊 Grafik Akurasi"):
                    st.line_chart(pd.DataFrame({"Akurasi (%)": akurasi_list}))
                with st.expander("🔥 Heatmap Akurasi per Digit"):
                    heat_df = pd.DataFrame({k: [sum(v)/len(v)*100 if v else 0] for k, v in digit_acc.items()})
                    fig, ax = plt.subplots()
                    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("⚠️ Tidak cukup data untuk evaluasi akurasi.")
