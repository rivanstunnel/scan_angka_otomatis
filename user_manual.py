import streamlit as st

def tampilkan_user_manual():
    with st.expander("📘 Panduan Pengguna Aplikasi", expanded=False):
        st.markdown("""
        ## 📖 PANDUAN PENGGUNA APLIKASI PREDIKSI 4D AI & MARKOV
        ---
        
        ### 1. Pengaturan Dasar
        - **🌍 Pilih Pasaran:** Tentukan wilayah pasaran angka (misal: SINGAPORE, SYDNEY, dll).
        - **📅 Pilih Hari:** Tentukan sumber data berdasarkan hari (harian, kemarin, hingga 5 hari lalu).
        - **📊 Data Uji Akurasi:** Tentukan berapa data historis terakhir yang digunakan untuk mengukur akurasi metode.
        - **🧠 Metode Prediksi:**
            - **Markov:** Model probabilistik sederhana.
            - **Markov Order-2:** Markov dengan 2 level urutan (lebih kompleks).
            - **Markov Gabungan:** Kombinasi Markov biasa dan Order-2.
            - **LSTM AI:** Model AI dengan arsitektur BiLSTM + Attention + Positional Encoding + Confidence.
            - **Ensemble AI + Markov:** Gabungan output dari model AI dan Markov untuk akurasi maksimal.
        
        ### 2. Cari Putaran Otomatis
        - **🔍 Toggle Aktif:** Sistem akan mencari jumlah putaran (n data terakhir) dengan akurasi prediksi tertinggi.
        - **Slider Manual:** Akan tetap muncul jika toggle dimatikan.
        - **✅ Otomatis Dipakai:** Jumlah putaran hasil pencarian akan langsung digunakan dalam API.

        ### 3. Pengambilan Data
        - Data akan diambil dari API dengan jumlah putaran dan hari yang telah dipilih.
        - **📥 Lihat Data:** Menampilkan angka hasil pengambilan untuk transparansi.

        ### 4. Manajemen Model AI (LSTM)
        - **📂 Status Model:** Menampilkan status model per digit (Digit-0 s/d Digit-3).
        - **📚 Latih & Simpan:** Melatih model jika belum tersedia, atau fine-tuning jika sudah ada.
        - **🗑 Hapus Model:** Menghapus model untuk digit tertentu agar bisa retrain dari awal.

        ### 5. Prediksi
        - **🔮 Tombol Prediksi:** Menghasilkan 6 angka Top-6 untuk setiap digit (ribuan, ratusan, puluhan, satuan).
        - **💡 Kombinasi 4D:** Jika menggunakan AI/Ensemble, akan muncul kombinasi lengkap dengan confidence score.
        - **⚡️ Confidence & Power:** Kombinasi disaring berdasarkan minimum confidence dan power weighting.

        ### 6. Evaluasi Akurasi
        - **📈 Grafik Akurasi:** Menampilkan hasil akurasi prediksi berdasarkan data uji.
        - **🔥 Heatmap Per Digit:** Akurasi per digit (ribuan, ratusan, dll) ditampilkan dalam bentuk heatmap.

        ### 7. Tips Akurasi Tinggi
        - Gunakan fitur **Cari Putaran Otomatis** untuk kestabilan hasil.
        - **Latih model** terlebih dahulu jika belum ada untuk pasaran tersebut.
        - Atur **minimum confidence** dan **power** agar kombinasi 4D yang muncul benar-benar kuat.
        - Gunakan metode **Ensemble AI + Markov** untuk hasil prediksi terbaik.

        ### 8. Catatan Tambahan
        - Model AI melatih dan menyimpan file per digit, dengan format `.h5` dan nama sesuai pasaran.
        - Anda bisa menyimpan, menghapus, dan memuat ulang model sesuai kebutuhan.
        - Sistem mendukung evaluasi otomatis saat prediksi dilakukan.

        ---
        🧠 *Aplikasi ini dirancang untuk eksplorasi AI prediksi angka berbasis data. Gunakan secara bijak dan bertanggung jawab.*
        """)