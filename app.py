import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from itertools import product
import streamlit_extras.stylable_container as stx # MODIFIKASI: Impor komponen alternatif

# Impor fungsi-fungsi dari file model
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

# Fungsi untuk tombol salin dari streamlit-extras
def copy_to_clipboard_button(text_to_copy, button_text, key):
    return stx.stylable_container(
        key=key,
        css_styles=f"""
            button {{
                background-color: #444454;
                color: white;
                border-radius: 10px;
            }}
            """,
        inner_html=f"<button onclick='navigator.clipboard.writeText(\"{text_to_copy}\")'>{button_text}</button>"
    )


lottie_predict = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")
st_lottie(lottie_predict, speed=1, height=150, key="prediksi")

st.title("🔮 Prediksi 4D - AI & Markov")

# Mendefinisikan list sebelum digunakan di dalam sidebar
hari_list = ["harian", "kemarin", "2hari", "3hari", "4hari", "5hari"]
metode_list = ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Pilih Hari", hari_list)
    putaran = st.number_input("🔁 Jumlah Putaran", min_value=1, max_value=1000, value=100)
    jumlah_uji = st.number_input("📊 Data Uji Akurasi", min_value=1, max_value=200, value=10)
    metode = st.selectbox("🧠 Metode Prediksi", metode_list)
    top_n = st.number_input("🔢 Jumlah Top Digit Prediksi", min_value=1, max_value=9, value=6)

    min_conf = 0.005
    power = 1.5
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        st.subheader("Pengaturan Lanjutan (AI)")
        min_conf = st.slider("🔎 Minimum Confidence", 0.0001, 0.001, 0.0005, step=0.0001, format="%.4f")
        power = st.slider("📈 Confidence Weight Power", 0.5, 3.0, 1.5, step=0.1)

    st.divider()
    if st.button("🔍 Cari Putaran Terbaik"):
        if 'df_data' not in st.session_state or len(st.session_state.df_data) < 30:
            st.warning("❌ Butuh minimal 30 data dari API untuk fitur ini.")
        else:
            st.session_state.run_best_putaran_search = True

# --- Ambil Data ---
query_id = f"{selected_lokasi}-{selected_hari}"
if 'df_data' not in st.session_state or st.session_state.get('last_query') != query_id:
    with st.spinner("🔄 Mengambil data dari API..."):
        try:
            url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran=1000&format=json&urut=asc"
            headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
            response = requests.get(url, headers=headers)
            data = response.json()
            all_angka = [item["result"] for item in data.get("data", []) if len(item["result"]) == 4 and item["result"].isdigit()]
            st.session_state.df_data = pd.DataFrame({"angka": all_angka})
            st.session_state.last_query = query_id
            st.success(f"✅ Total {len(st.session_state.df_data)} data berhasil diambil dari API.")
        except Exception as e:
            st.error(f"❌ Gagal ambil data API: {e}")
            st.session_state.df_data = pd.DataFrame({"angka": []})

df = st.session_state.get('df_data', pd.DataFrame()).tail(putaran)
with st.expander(f"📥 Menampilkan {len(df)} dari {st.session_state.get('df_data', pd.DataFrame()).shape[0]} data terakhir"):
    st.code("\n".join(df['angka'].tolist()), language="text")

# Tombol Prediksi Utama
if st.button("🔮 Prediksi Sekarang!", use_container_width=True):
    st.session_state.run_best_putaran_search = False
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

        if result is None:
            st.error("❌ Gagal melakukan prediksi. Pastikan model AI sudah dilatih jika menggunakan metode AI.")
        else:
            st.subheader(f"🎯 Hasil Prediksi Top {top_n} Digit")
            labels = ["As", "Kop", "Kepala", "Ekor"]
            for i, label in enumerate(labels):
                st.markdown(f"#### **{label}:** {', '.join(map(str, result[i]))}")

            st.divider()
            with st.expander("⬇️ Tampilkan & Salin Hasil Kombinasi"):
                kombinasi_4d_list = ["".join(map(str, p)) for p in product(result[0], result[1], result[2], result[3])]
                kombinasi_3d_list = ["".join(map(str, p)) for p in product(result[1], result[2], result[3])]
                kombinasi_2d_list = ["".join(map(str, p)) for p in product(result[2], result[3])]
                
                separator = " * "
                text_4d = separator.join(kombinasi_4d_list)
                text_3d = separator.join(kombinasi_3d_list)
                text_2d = separator.join(kombinasi_2d_list)

                tab4d, tab3d, tab2d = st.tabs([f"Kombinasi 4D ({len(kombinasi_4d_list)})", f"Kombinasi 3D ({len(kombinasi_3d_list)})", f"Kombinasi 2D ({len(kombinasi_2d_list)})"])

                # --- MODIFIKASI: Menggunakan komponen dari streamlit-extras ---
                with tab4d:
                    st.text_area("Hasil 4D (As-Kop-Kepala-Ekor)", text_4d, height=200, key="text_4d")
                    copy_to_clipboard_button(text_4d.replace('"', '\\"'), "📋 Salin Hasil 4D", key="copy4d")


                with tab3d:
                    st.text_area("Hasil 3D (Kop-Kepala-Ekor)", text_3d, height=200, key="text_3d")
                    copy_to_clipboard_button(text_3d.replace('"', '\\"'), "📋 Salin Hasil 3D", key="copy3d")

                with tab2d:
                    st.text_area("Hasil 2D (Kepala-Ekor)", text_2d, height=200, key="text_2d")
                    copy_to_clipboard_button(text_2d.replace('"', '\\"'), "📋 Salin Hasil 2D", key="copy2d")
                # --- AKHIR MODIFIKASI ---

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                # ... (Sisa kode tidak berubah)
                pass

        # ... (Sisa kode tidak berubah)
        # ... (Pastikan sisa kode Anda untuk evaluasi akurasi tetap ada di sini)
        pass

# ... (Sisa kode tidak berubah)
# ... (Pastikan sisa kode Anda untuk pencarian putaran terbaik tetap ada di sini)
pass
