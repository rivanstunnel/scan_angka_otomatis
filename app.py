# --- BAGIAN ANALISIS PUTARAN TERBAIK (DIUBAH) ---
if st.session_state.get('run_putaran_analysis', False):
    st.header("🔬 Hasil Analisis Putaran Terbaik")
    with st.spinner("Menganalisis berbagai jumlah putaran... Ini akan memakan waktu."):
        full_df = st.session_state.get('df_data', pd.DataFrame())
        putaran_results = {}
        # Menghitung jumlah maksimum putaran yang mungkin berdasarkan data yang tersedia
        max_putaran_test = len(full_df) - jumlah_uji

        start_putaran = 11
        # PERBAIKAN: Hapus batas hardcode 1000, gunakan batas maksimal dari data
        end_putaran = max_putaran_test
        
        step_putaran = 1

        if end_putaran < start_putaran:
            st.warning(f"Data tidak cukup untuk pengujian. Dibutuhkan setidaknya {start_putaran + jumlah_uji} total data riwayat.")
        else:
            test_range = list(range(start_putaran, end_putaran + 1, step_putaran))
            
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
                m2.metric("Akurasi Tertinggi", f"{best_accuracy:.2f}%", f"Dengan {best_putaran} data")

                chart_data = pd.DataFrame.from_dict(putaran_results, orient='index', columns=['Akurasi (%)'])
                chart_data.index.name = 'Jumlah Putaran'
                st.line_chart(chart_data)

                st.subheader(f"📜 Tabel Hasil Analisis Putaran (Rentang {start_putaran}-{end_putaran})")
                sorted_chart_data = chart_data.sort_values(by='Akurasi (%)', ascending=False)
                sorted_chart_data['Akurasi (%)'] = sorted_chart_data['Akurasi (%)'].map('{:.2f}%'.format)
                st.dataframe(sorted_chart_data, use_container_width=True)

    st.session_state.run_putaran_analysis = False
