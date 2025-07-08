import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
from itertools import product
from markov_model import top6_markov

class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return x + tf.cast(pos_encoding, tf.float32)

def preprocess_data(df, window_size=5):
    if len(df) < window_size + 1:
        return np.array([]), [np.array([]) for _ in range(4)]
    sequences, targets = [], [[] for _ in range(4)]
    angka = df["angka"].values
    for i in range(len(angka) - window_size):
        window = angka[i:i+window_size]
        if any(len(x) != 4 or not x.isdigit() for x in window):
            continue
        seq = [int(d) for num in window[:-1] for d in f"{int(num):04d}"]
        sequences.append(seq)
        target_digits = [int(d) for d in f"{int(window[-1]):04d}"]
        for j in range(4):
            targets[j].append(to_categorical(target_digits[j], num_classes=10))
    X = np.array(sequences)
    y = [np.array(t) for t in targets]
    return X, y

def build_lstm_model(input_len, embed_dim=32, lstm_units=128, attention_heads=4, temperature=0.5):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embed_dim)(inputs)
    x = PositionalEncoding()(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=attention_heads, key_dim=embed_dim)(x, x)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_transformer_model(input_len, embed_dim=32, heads=4, temperature=0.5):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embed_dim)(inputs)
    x = PositionalEncoding()(x)
    for _ in range(2):
        attn = MultiHeadAttention(num_heads=heads, key_dim=embed_dim)(x, x)
        x = LayerNormalization()(x + attn)
        ff = Dense(embed_dim, activation='relu')(x)
        x = LayerNormalization()(x + ff)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_model(df, lokasi, window_size=5, model_type="lstm"):
    if len(df) < window_size + 5:
        return
    X, y_all = preprocess_data(df, window_size=window_size)
    if X.shape[0] == 0:
        return
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    for i in range(4):
        y = y_all[i]
        model = (
            build_transformer_model(X.shape[1]) if model_type == "transformer"
            else build_lstm_model(X.shape[1])
        )
        suffix = f"{model_type}"
        log_path = f"training_logs/history_{lokasi.lower().replace(' ', '_')}_digit{i}_{suffix}.csv"
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.2, callbacks=callbacks)
        model.save(f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}_{suffix}.h5")

def model_exists(lokasi, model_type="lstm"):
    return all(os.path.exists(f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}_{model_type}.h5") for i in range(4))

def top6_model(df, lokasi=None, model_type="lstm", return_probs=False, temperature=0.5):
    X, _ = preprocess_data(df)
    if X.shape[0] == 0:
        return None
    results, probs = [], []
    for i in range(4):
        path = f"saved_models/{lokasi.lower().replace(' ', '_')}_digit{i}_{model_type}.h5"
        if not os.path.exists(path):
            return None
        try:
            model = load_model(path, compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})
            if model.input_shape[1] != X.shape[1]:
                return None
            pred = model.predict(X, verbose=0)
            avg = np.mean(pred, axis=0)
            avg /= avg.sum()  # Confidence normalization
            top6 = avg.argsort()[-6:][::-1]
            results.append(list(top6))
            probs.append(avg[top6])
        except Exception as e:
            print(f"[{model_type.upper()} ERROR digit {i}] {e}")
            return None
    return (results, probs) if return_probs else results

def kombinasi_4d(df, lokasi, model_type="lstm", top_n=10, min_conf=0.0001, power=1.5, mode='product'):
    result, probs = top6_model(df, lokasi=lokasi, model_type=model_type, return_probs=True)
    if result is None or probs is None:
        return []
    combinations = list(product(*result))
    scores = []
    for combo in combinations:
        digit_scores = []
        valid = True
        for i in range(4):
            try:
                idx = result[i].index(combo[i])
                digit_scores.append(probs[i][idx] ** power)
            except:
                valid = False
                break
        if not valid:
            continue
        score = np.prod(digit_scores) if mode == 'product' else np.mean(digit_scores)
        if score >= min_conf:
            scores.append(("".join(map(str, combo)), score))
    topk = sorted(scores, key=lambda x: -x[1])[:top_n]
    return topk

def top6_ensemble(df, lokasi, model_type="lstm", lstm_weight=0.6, markov_weight=0.4):
    lstm_result = top6_model(df, lokasi=lokasi, model_type=model_type)
    markov_result, _ = top6_markov(df)
    if lstm_result is None or markov_result is None:
        return None
    ensemble = []
    for i in range(4):
        all_digits = lstm_result[i] + markov_result[i]
        scores = {}
        for digit in all_digits:
            scores[digit] = scores.get(digit, 0)
            if digit in lstm_result[i]:
                scores[digit] += lstm_weight
            if digit in markov_result[i]:
                scores[digit] += markov_weight
        top6 = sorted(scores.items(), key=lambda x: -x[1])[:6]
        ensemble.append([x[0] for x in top6])
    return ensemble
