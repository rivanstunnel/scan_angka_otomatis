# Simpan sebagai ai_model.py

import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from itertools import product

from markov_model import top7_markov_hybrid # Impor versi top7

# Helper function (sama seperti di markov_model.py)
def _prepare_data(df):
    if 'angka' not in df.columns or df.empty:
        return None
    data = df['angka'].astype(str).str.zfill(4)
    digits = pd.DataFrame({
        'd1': data.str[0].astype(int),
        'd2': data.str[1].astype(int),
        'd3': data.str[2].astype(int),
        'd4': data.str[3].astype(int),
    })
    return digits

# Helper untuk membuat sekuens data untuk LSTM
def _create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def model_exists(lokasi):
    """Cek apakah keempat model untuk lokasi tertentu sudah ada."""
    for i in range(4):
        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        if not os.path.exists(model_path):
            return False
    return True

# Fungsi ini tidak perlu diubah
def train_and_save_lstm(df, lokasi, seq_length=10):
    digits = _prepare_data(df)
    if digits is None or len(digits) <= seq_length:
        print("Data tidak cukup untuk training.")
        return

    base_path = "saved_models"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for i in range(1, 5):
        col = f'd{i}'
        X, y = _create_sequences(digits[col].values, seq_length)
        if X.shape[0] == 0: continue

        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # One-hot encode y
        y_onehot = np.zeros((y.shape[0], 10))
        y_onehot[np.arange(y.shape[0]), y] = 1

        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1)),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(X, y_onehot, epochs=50, batch_size=1, verbose=0)
        
        model_name = f"{lokasi.lower().replace(' ', '_')}_digit{i-1}.h5"
        model.save(os.path.join(base_path, model_name))

# MODIFIKASI UTAMA: Mengambil 7 prediksi teratas
def top7_lstm(df, lokasi, seq_length=10, n=7):
    if not model_exists(lokasi):
        return None
        
    digits = _prepare_data(df)
    if digits is None or len(digits) < seq_length:
        return None

    predictions = []
    for i in range(4):
        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        model = load_model(model_path)
        
        last_sequence = digits[f'd{i+1}'].values[-seq_length:]
        input_seq = last_sequence.reshape((1, seq_length, 1))
        
        probabilities = model.predict(input_seq, verbose=0)[0]
        # Ambil indeks dari 7 probabilitas tertinggi
        top_n_indices = np.argsort(probabilities)[-n:][::-1]
        predictions.append(top_n_indices.tolist())
        
    return predictions

# MODIFIKASI UTAMA: Menggabungkan hasil dari model top7
def top7_ensemble(df, lokasi, seq_length=10, n=7):
    # Dapatkan prediksi dari LSTM
    lstm_preds = top7_lstm(df, lokasi, seq_length, n)
    if lstm_preds is None:
        return None # Gagal jika model LSTM tidak ada

    # Dapatkan prediksi dari Markov Hybrid
    markov_preds = top7_markov_hybrid(df, n)
    if markov_preds is None:
        return lstm_preds # Fallback ke LSTM jika Markov gagal
    
    ensemble_preds = []
    for i in range(4):
        # Gabungkan hasil dan ambil yang unik, prioritaskan LSTM
        combined = list(dict.fromkeys(lstm_preds[i] + markov_preds[i]))
        ensemble_preds.append(combined[:n])

    return ensemble_preds

# Fungsi ini tidak perlu diubah, karena bekerja berdasarkan probabilitas,
# bukan pada jumlah digit yang diprediksi.
def kombinasi_4d(df, lokasi, top_n=10, seq_length=10, min_conf=0.0005, power=1.5):
    if not model_exists(lokasi) or len(_prepare_data(df)) < seq_length:
        return []

    all_probs = []
    digits = _prepare_data(df)
    for i in range(4):
        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        model = load_model(model_path)
        last_sequence = digits[f'd{i+1}'].values[-seq_length:]
        input_seq = last_sequence.reshape((1, seq_length, 1))
        probabilities = model.predict(input_seq, verbose=0)[0]
        all_probs.append(probabilities)

    digit_options = [range(10)] * 4
    all_kombinasi = product(*digit_options)
    
    scored_kombinasi = []
    for komb in all_kombinasi:
        score = 1.0
        valid = True
        for i in range(4):
            digit_prob = all_probs[i][komb[i]]
            if digit_prob < min_conf:
                valid = False
                break
            score *= digit_prob
        
        if valid:
            komb_str = "".join(map(str, komb))
            # Terapkan power weighting untuk menonjolkan kombinasi dengan probabilitas tinggi
            final_score = score ** power 
            scored_kombinasi.append((komb_str, final_score))

    scored_kombinasi.sort(key=lambda x: x[1], reverse=True)
    return scored_kombinasi[:top_n]
