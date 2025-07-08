# ai_model.py

import numpy as np
import pandas as pd
import os
from itertools import product

# Impor library AI, bungkus dalam try-except untuk menghindari error jika tidak terinstal
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("Peringatan: TensorFlow tidak terinstal. Fungsi AI tidak akan berjalan.")
    Sequential = None # Jadikan None jika gagal impor

# --- Variabel Global & Fungsi Pembantu ---
MODEL_DIR = "saved_models"
SEQUENCE_LEN = 10 # Panjang sekuens data historis untuk memprediksi data berikutnya

def _get_digits(df):
    """Mengubah dataframe angka string menjadi list of lists of int."""
    if df.empty:
        return np.array([])
    return np.array(df['angka'].apply(lambda x: [int(d) for d in str(x).zfill(4)]).tolist())

def _create_sequences(data, seq_len=SEQUENCE_LEN):
    """Membuat sekuens data untuk input LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def model_exists(lokasi):
    """Cek apakah semua 4 model untuk suatu lokasi sudah ada."""
    for i in range(4):
        model_path = os.path.join(MODEL_DIR, f"{lokasi.lower().replace(' ', '_')}_digit{i}.h5")
        if not os.path.exists(model_path):
            return False
    return True

# --- Fungsi Manajemen Model ---
def train_and_save_lstm(df, lokasi):
    """Melatih satu model LSTM per digit dan menyimpannya."""
    if Sequential is None:
        raise ImportError("TensorFlow tidak terinstal. Silakan install dengan 'pip install tensorflow'")

    digits = _get_digits(df)
    if digits.shape[0] < SEQUENCE_LEN + 1:
        print(f"Data tidak cukup untuk melatih. Butuh {SEQUENCE_LEN + 1}, tersedia {digits.shape[0]}")
        return

    for i in range(4): # Loop untuk setiap posisi digit (Ribuan, Ratusan, Puluhan, Satuan)
        pos_data = digits[:, i]
        X, y = _create_sequences(pos_data)
        
        if X.shape[0] == 0: continue # Lewati jika tidak ada sekuens yang bisa dibuat
        
        # Reshape input untuk LSTM [samples, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Buat arsitektur model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(SEQUENCE_LEN, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax') # 10 output untuk digit 0-9
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=[early_stopping])
        
        # Simpan model
        model_path = os.path.join(MODEL_DIR, f"{lokasi.lower().replace(' ', '_')}_digit{i}.h5")
        model.save(model_path)


def _predict_lstm_probs(df, lokasi):
    """Fungsi internal untuk mendapatkan probabilitas prediksi dari model LSTM."""
    if not model_exists(lokasi): return None

    digits = _get_digits(df)
    all_probs = []

    for i in range(4):
        model_path = os.path.join(MODEL_DIR, f"{lokasi.lower().replace(' ', '_')}_digit{i}.h5")
        model = load_model(model_path, compile=False)
        
        pos_data = digits[:, i]
        last_sequence = pos_data[-SEQUENCE_LEN:].reshape(1, SEQUENCE_LEN, 1)
        
        probs = model.predict(last_sequence)[0]
        all_probs.append(probs)
        
    return np.array(all_probs)


# --- Fungsi Prediksi Utama ---
def predict_lstm(df, lokasi, top_n=6):
    """Melakukan prediksi menggunakan model LSTM yang sudah ada."""
    all_probs = _predict_lstm_probs(df, lokasi)
    if all_probs is None: return None
    
    # Ambil N digit teratas berdasarkan probabilitas
    result = [np.argsort(probs)[-top_n:][::-1] for probs in all_probs]
    return result

def predict_ensemble(df, lokasi, top_n=6):
    """Menggabungkan prediksi dari Markov Hybrid dan LSTM."""
    from markov_model import predict_markov_hybrid # Impor di sini untuk menghindari circular import

    # Dapatkan probabilitas dari kedua model
    probs_lstm = _predict_lstm_probs(df, lokasi)
    
    # Ambil probabilitas dari markov hybrid
    digits = _get_digits(df)
    if digits.shape[0] < 2: return None
    _, probs_markov_o1 = predict_markov_hybrid(df, top_n=10) # Menggunakan hybrid sebagai representasi markov

    if probs_lstm is None or probs_markov_o1 is None:
        print("Gagal mendapatkan prediksi dari salah satu model, ensemble dibatalkan.")
        return None

    # Beri bobot: 60% LSTM, 40% Markov
    ensemble_probs = (0.6 * probs_lstm) + (0.4 * probs_markov_o1)
    
    result = [np.argsort(probs)[-top_n:][::-1] for probs in ensemble_probs]
    return result


def kombinasi_4d(df, lokasi, top_n=10, min_conf=0.001, power=1.5):
    """Menghitung rekomendasi kombinasi 4D berdasarkan confidence dari LSTM."""
    all_probs = _predict_lstm_probs(df, lokasi)
    if all_probs is None: return []

    # Ambil top 4 digit dari setiap posisi untuk membuat kombinasi
    top_digits_per_pos = [np.argsort(probs)[-4:][::-1] for probs in all_probs]
    
    combinations = []
    for p in product(*top_digits_per_pos):
        komb = "".join(map(str, p))
        
        # Hitung skor confidence dengan bobot
        score = (
            all_probs[0][p[0]] ** power *
            all_probs[1][p[1]] ** power *
            all_probs[2][p[2]] ** power *
            all_probs[3][p[3]] ** power
        )
        
        if score >= min_conf:
            combinations.append((komb, score))
            
    # Urutkan berdasarkan skor tertinggi
    combinations.sort(key=lambda x: x[1], reverse=True)
    
    return combinations[:top_n]
