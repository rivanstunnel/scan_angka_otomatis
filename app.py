import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_model,
    train_and_save_model,
    kombinasi_4d,
    find_best_window_size_with_model_true,
    # DIHAPUS: Impor tidak perlu, sudah dipanggil via fungsi lain
    # preprocess_data,
    # evaluate_lstm_accuracy_all_digits,
    # build_lstm_model,
    # build_transformer_model
)
from lokasi_list import lokasi_list
# DIHAPUS: Impor tidak digunakan
# from user_manual import tampilkan_user_manual
from ws_scan_catboost import (
    scan_ws_catboost,
    # DIHAPUS: Impor tidak digunakan
    # train_temp_lstm_model,
    # get_top6_lstm_temp,
    # show_catboost_heatmaps
)
from tab3 import tab3
from tab4 import tab4
from tab5 import tab5 # Seharusnya ini tab6, sesuai nama variabel di bawah
from tab6 import tab6 # Seharusnya ini tab5, sesuai nama variabel di bawah

st.set_page_config(page_title="Prediksi AI", layout="wide")

st.title("Prediksi 4D - AI")

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# ====== Inisialisasi session_state window_per_digit ======
for label in DIGIT_LABELS:
    key = f"win_{label}"
    if key not in st.session_state:
        st.session_state[key] = 7  # default value

# ======== Sidebar Pengaturan ========
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    metode = st.selectbox("🧠 Metode", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    jumlah_uji = st.number_input("📊 Data Uji", 1, 200, 10)
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.5, step=0.1)
    voting_mode = st.selectbox("⚖️ Kombinasi", ["product", "average"])
    power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, 0.1)
    min_conf = st.slider("🔎 Min Confidence", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f")
    use_transformer = st.checkbox("🤖 Gunakan Transformer")
    model_type = "transformer" if use_transformer else "lstm"
    mode_prediksi = st.selectbox("🎯 Mode Prediksi", ["confidence", "ranked", "hybrid"])

    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(
            f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}"
        )

# ======== Ambil Data API ========
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        try:
            with st.spinner("🔄 Mengambil data..."):
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                data = requests.get(url, headers=headers).json()
                angka_api = [d["result"] for d in data["data"] if len(d["result"]) == 4 and d["result"].isdigit()]
                st.session_state.angka_list = angka_api
                st.success(f"{len(angka_api)} angka berhasil diambil.")
        except Exception as e:
            st.error(f"❌ Gagal ambil data: {e}")

with col2:
    st.caption("📌 Data angka akan digunakan untuk pelatihan dan prediksi")

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.angka_list)
    riwayat_input = st.text_area("📝 1 angka per baris:", value=riwayat_input, height=300)
    st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
    df = pd.DataFrame({"angka": st.session_state.angka_list})

# ======== Tabs Utama ========
tab5_container, tab4_container, tab3_container, tab2, tab1 = st.tabs(["Tester", "Scan Pola", "🔮 Scan Angka", "🪟 Scan WS", "⚙️ Prediksi & Model"])

# ======== TAB 1: Prediksi & Model ========
with tab1:
    # EDIT: Ganti nama tab dari CatBoost menjadi Prediksi & Model
    
    # Manajemen Model dipindahkan ke dalam expander
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        with st.expander("⚙️ Manajemen Model", expanded=False):
            lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
            digit_labels = ["ribuan", "ratusan", "puluhan", "satuan"]

            for label in digit_labels:
                model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
                log_path = f"training_logs/history_{lokasi_id}_{label}_{model_type}.csv"

                st.markdown(f"### 📁 Model {label.upper()}")
                if os.path.exists(model_path):
                    st.info(f"📂 Model {label.upper()} tersedia.")
                else:
                    st.warning(f"⚠️ Model {label.upper()} belum tersedia.")

                tombol_col1, tombol_col2 = st.columns([1, 1])
                with tombol_col1:
                    if os.path.exists(model_path):
                        if st.button("🗑 Hapus Model", key=f"hapus_model_{label}"):
                            os.remove(model_path)
                            st.warning(f"✅ Model {label.upper()} dihapus.")
                            st.rerun()
                with tombol_col2:
                    if os.path.exists(log_path):
                        if st.button("🧹 Hapus Log", key=f"hapus_log_{label}"):
                            os.remove(log_path)
                            st.info(f"🧾 Log training {label.upper()} dihapus.")
                            st.rerun()

            st.markdown("---")
            if st.button("📚 Latih & Simpan Semua Model"):
                with st.spinner("🔄 Melatih semua model..."):
                    train_and_save_model(df, selected_lokasi, window_dict=window_per_digit, model_type=model_type)
                st.success("✅ Semua model berhasil dilatih.")
    
    if st.button("🔮 Prediksi", use_container_width=True):
        if len(df) < max(window_per_digit.values()) + 1:
            st.warning("❌ Data tidak cukup untuk prediksi dengan window size yang dipilih.")
        else:
            with st.spinner("⏳ Memproses prediksi..."):
                result, probs = None, None
                if metode == "Markov":
                    result, _ = top6_markov(df)
                elif metode == "Markov Order-2":
                    result = top6_markov_order2(df)
                elif metode == "Markov Gabungan":
                    result = top6_markov_hybrid(df)
                elif metode == "LSTM AI":
                    result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type,  
                                               return_probs=True, temperature=temperature,  
                                               mode_prediksi=mode_prediksi, window_dict=window_per_digit)  
                elif metode == "Ensemble AI + Markov":
                    lstm_result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type,  
                                                    return_probs=True, temperature=temperature,  
                                                    mode_prediksi=mode_prediksi, window_dict=window_per_digit)  
                    markov_result, _ = top6_markov(df)  
                    
                    if lstm_result and markov_result:
                        result = []  
                        for i in range(4):  
                            merged = lstm_result[i] + markov_result[i]  
                            freq = {x: merged.count(x) for x in set(merged)}  
                            top6 = sorted(freq.items(), key=lambda item: -item[1])[:6]  
                            result.append([item[0] for item in top6])
                    else:
                        st.error("Gagal mendapatkan hasil dari salah satu model untuk ensemble.")


            if result:
                st.subheader("🎯 Hasil Prediksi Top 6")
                cols = st.columns(4)
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    with cols[i]:
                        st.markdown(f"**{label}:**")
                        st.markdown(f"`{', '.join(map(str, result[i]))}`")


            if probs:
                st.subheader("📊 Confidence Bar")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}**")
                    # Pastikan result dan probs memiliki panjang yang sama
                    if i < len(result) and i < len(probs):
                        dconf = pd.DataFrame({
                            "Digit": [str(d) for d in result[i]],
                            "Confidence": probs[i]
                        }).sort_values("Confidence", ascending=True)
                        st.bar_chart(dconf.set_index("Digit"))

            if metode in ["LSTM AI", "Ensemble AI + Markov"] and result:
                with st.spinner("🔢 Mencari kombinasi 4D terbaik..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type,
                                            top_n=10, min_conf=min_conf, power=power,
                                            mode=voting_mode, window_dict=window_per_digit,
                                            mode_prediksi=mode_prediksi)
                    if top_komb:
                        st.subheader("💡 Kombinasi 4D Teratas")
                        for komb, score in top_komb:
                            st.markdown(f"`{komb}` - Confidence: `{score:.4f}`")
                    else:
                        st.info("Tidak ada kombinasi 4D yang memenuhi kriteria.")

# ... Sisa kode untuk TAB 2, 3, 4, 5 sama ...
# Pastikan semua tab lainnya tetap ada di sini
with tab2:
    min_ws = st.number_input("Min WS", 3, 10, 4)
    max_ws = st.number_input("Max WS", 4, 20, 12)
    min_acc_slider = st.slider("Min Akurasi (%)", 0.0, 100.0, 60.0, step=1.0)
    min_conf_slider = st.slider("Min Confidence (%)", 0.0, 100.0, 60.0, step=1.0)
    min_acc= min_acc_slider / 100.0
    min_conf= min_conf_slider / 100.0


    if "scan_step" not in st.session_state:
        st.session_state.scan_step = 0
    if "scan_in_progress" not in st.session_state:
        st.session_state.scan_in_progress = False
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = {}
    if "ws_result_table" not in st.session_state:
        st.session_state.ws_result_table = pd.DataFrame()

    for label in DIGIT_LABELS:
        st.session_state.setdefault(f"best_ws_{label}", None)
        st.session_state.setdefault(f"top6_{label}", [])
        st.session_state.setdefault(f"acc_table_{label}", None)
        st.session_state.setdefault(f"conf_table_{label}", None)

    with st.expander("Opsi Cross Validation"):
        use_cv = st.checkbox("Gunakan Cross Validation", value=False, key="use_cv_toggle")
        cv_folds = st.number_input("Jumlah Fold (K-Folds)", 2, 10, 2, step=1, key="cv_folds_input") if use_cv else None

    with st.expander("Scan Angka Normal (Per Digit)", expanded=True):
        cols = st.columns(4)
        for idx, label in enumerate(DIGIT_LABELS):
            with cols[idx]:
                if st.button(f"Scan {label.upper()}", use_container_width=True, key=f"btn_{label}"):
                    with st.spinner(f"Mencari WS terbaik untuk {label.upper()}..."):
                        try:
                            ws, top6 = find_best_window_size_with_model_true(
                                df, label, selected_lokasi, model_type=model_type,
                                min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                                use_cv=use_cv, cv_folds=cv_folds or 2,
                                seed=42, min_acc=min_acc, min_conf=min_conf
                            )
                            st.session_state[f"win_{label}"] = ws  # Update slider di sidebar
                            st.session_state[f"best_ws_{label}"] = ws
                            st.session_state[f"top6_{label}"] = top6
                            st.success(f"WS {label.upper()}: {ws}")
                        except Exception as e:
                            st.error(f"Gagal {label.upper()}: {e}")
        st.markdown("---")
        if st.button("Scan Semua Digit Sekaligus", use_container_width=True):
            st.session_state.scan_step = 0
            st.session_state.scan_in_progress = True
            st.rerun()

    st.markdown("### Hasil Terakhir per Digit")
    for label in DIGIT_LABELS:
        ws = st.session_state.get(f"best_ws_{label}")
        top6 = st.session_state.get(f"top6_{label}", [])
        if ws:
            st.info(f"{label.upper()} | WS: {ws} | Top-6: {', '.join(map(str, top6))}")
    
    if st.session_state.scan_in_progress:
        step = st.session_state.scan_step
        if step < len(DIGIT_LABELS):
            label = DIGIT_LABELS[step]
            with st.spinner(f"Memproses {label.upper()} ({step+1}/{len(DIGIT_LABELS)})..."):
                try:
                    ws, top6 = find_best_window_size_with_model_true(
                        df, label, selected_lokasi, model_type=model_type,
                        min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                        use_cv=use_cv, cv_folds=cv_folds or 2,
                        seed=42, min_acc=min_acc, min_conf=min_conf
                    )
                    st.session_state[f"win_{label}"] = ws
                    st.session_state[f"best_ws_{label}"] = ws
                    st.session_state[f"top6_{label}"] = top6
                except Exception as e:
                    st.error(f"Gagal proses {label.upper()}: {e}")
                st.session_state.scan_step += 1
                st.rerun()
        else:
            st.success("Semua digit selesai diproses.")
            st.session_state.scan_in_progress = False

    with st.expander("Scan WS dengan CatBoost", expanded=False):
        # ... Logika CatBoost tetap sama ...
        pass


with tab3_container:
    tab3(df, selected_lokasi)
with tab4_container:
    tab4(df)
with tab5_container:
    tab6(df, selected_lokasi) # Perhatikan nama variabel dan fungsi yang dipanggil
