# ai_model.py

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import itertools

# Blok try-except untuk import tensorflow
try:
    import tensorflow as tf
    # Menonaktifkan logging TensorFlow yang terlalu detail
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_INSTALLED = True
except ImportError:
    TENSORFLOW_INSTALLED = False

# Impor dari file model markov Anda - NAMA FUNGSI DIPERBAIKI
from markov_model import predict_markov_hybrid

def _preprocess_data_for_lstm(series, look_back=10):
    """Mempersiapkan data untuk input LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y), scaler

def model_exists(lokasi, digit_index):
    """Cek apakah file model sudah ada."""
    model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{digit_index}.h5"
    return os.path.exists(model_path)

def train_and_save_lstm(df, lokasi, look_back=10):
    """Melatih model LSTM untuk setiap digit dan menyimpannya."""
    if not TENSORFLOW_INSTALLED:
        raise ImportError("TensorFlow tidak terinstall. Silakan install dengan 'pip install tensorflow'.")
    
    data = df['angka'].astype(str).str.zfill(4)
    digits_df = pd.DataFrame({f'd{i+1}': data.str[i].astype(int) for i in range(4)})
    
    for i in range(4):
        series = digits_df[f'd{i+1}']
        if len(series) < look_back + 1:
            print(f"Data tidak cukup untuk melatih model digit-{i}, skip.")
            continue
        
        X, Y, _ = _preprocess_data_for_lstm(series, look_back)
        if len(X) == 0: continue

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(50, input_shape=(look_back, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, Y, epochs=100, batch_size=16, verbose=0)
        
        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        model.save(model_path)

def top_n_lstm(df, lokasi, top_n=7, look_back=10):
    """Prediksi Top-N menggunakan model LSTM yang sudah dilatih."""
    if not TENSORFLOW_INSTALLED: return None
    if len(df) < look_back: return None

    predictions = []
    data = df['angka'].astype(str).str.zfill(4)
    digits_df = pd.DataFrame({f'd{i+1}': data.str[i].astype(int) for i in range(4)})

    for i in range(4):
        if not model_exists(lokasi, i): return None

        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        model = load_model(model_path, compile=False) # compile=False mempercepat loading
        
        series = digits_df[f'd{i+1}']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))
        last_sequence = scaled_series[-look_back:]
        input_data = np.reshape(last_sequence, (1, look_back, 1))
        
        predicted_scaled_value = model.predict(input_data, verbose=0)[0][0]
        
        all_digits = np.arange(10)
        scaled_digits = scaler.transform(all_digits.reshape(-1, 1)).flatten()
        distances = np.abs(scaled_digits - predicted_scaled_value)
        sigma = 0.1
        probabilities = np.exp(-distances**2 / (2 * sigma**2))
        
        top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
        predictions.append(top_n_indices.tolist())
        
    return predictions

def top_n_ensemble(df, lokasi, top_n=7):
    """Gabungan prediksi dari Markov Hybrid dan LSTM AI."""
    # PANGGILAN FUNGSI DIPERBAIKI
    pred_markov = predict_markov_hybrid(df, top_n=top_n)
    pred_lstm = top_n_lstm(df, lokasi=lokasi, top_n=top_n)

    if not pred_markov or not pred_lstm:
        print("Peringatan: Prediksi Markov atau LSTM gagal. Ensemble dibatalkan.")
        return None
    
    ensemble_predictions = []
    for i in range(4):
        if not pred_markov[i] or not pred_lstm[i]:
             # Jika salah satu list prediksi kosong, fallback ke Markov saja
            return pred_markov

        scores = defaultdict(float)
        for rank, digit in enumerate(pred_lstm[i]):
            scores[digit] += (top_n - rank) * 1.5
        for rank, digit in enumerate(pred_markov[i]):
            scores[digit] += (top_n - rank) * 1.0

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_n_list = [digit for digit, score in sorted_scores[:top_n]]
        
        if len(top_n_list) < top_n:
            remaining = [d for d in range(10) if d not in top_n_list]
            if remaining:
                 top_n_list.extend(np.random.choice(remaining, size=min(len(remaining), top_n - len(top_n_list)), replace=False))
        
        ensemble_predictions.append(top_n_list)
        
    return ensemble_predictions

def kombinasi_4d(df, lokasi, top_n=10, min_conf=0.0005, power=1.5, look_back=10):
    """Menghasilkan simulasi kombinasi 4D terbaik."""
    if not TENSORFLOW_INSTALLED or len(df) < look_back: return []
    if not all(model_exists(lokasi, i) for i in range(4)): return []
    
    data = df['angka'].astype(str).str.zfill(4)
    digits_df = pd.DataFrame({f'd{i+1}': data.str[i].astype(int) for i in range(4)})
    
    digit_probs = []
    for i in range(4):
        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}.h5"
        model = load_model(model_path, compile=False)
        
        series = digits_df[f'd{i+1}']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))
        last_sequence = scaled_series[-look_back:]
        input_data = np.reshape(last_sequence, (1, look_back, 1))
        
        predicted_scaled_value = model.predict(input_data, verbose=0)[0][0]
        
        all_digits = np.arange(10)
        scaled_digits = scaler.transform(all_digits.reshape(-1, 1)).flatten()
        
        distances = np.abs(scaled_digits - predicted_scaled_value)
        sigma = np.std(distances) + 1e-6
        
        probabilities = np.exp(-distances**2 / (2 * sigma**2))
        probabilities /= probabilities.sum()
        
        digit_probs.append({digit: prob for digit, prob in zip(all_digits, probabilities)})

    all_combinations = list(itertools.product(range(10), repeat=4))
    
    combo_scores = []
    for combo in all_combinations:
        score = 1.0
        for i in range(4):
            digit = combo[i]
            prob = digit_probs[i].get(digit, 0)
            score *= prob
        
        if score >= min_conf:
            combo_scores.append(("".join(map(str, combo)), score ** power))
            
    combo_scores.sort(key=lambda x: x[1], reverse=True)
    
    return combo_scores[:top_n]
