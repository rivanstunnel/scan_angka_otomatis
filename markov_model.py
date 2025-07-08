# markov_model.py

import pandas as pd
import numpy as np

def _get_digits(df):
    """Membantu mengubah dataframe angka string menjadi list of lists of int."""
    if df.empty:
        return np.array([])
    return np.array(df['angka'].apply(lambda x: [int(d) for d in str(x).zfill(4)]).tolist())

def predict_markov(df, top_n=6):
    """
    Prediksi berdasarkan transisi dari satu digit ke digit berikutnya (Order-1).
    Contoh: P(Ratusan | Ribuan), P(Puluhan | Ratusan), dst.
    """
    digits = _get_digits(df)
    if digits.shape[0] < 1:
        return None, None

    # Matriks transisi: [posisi_digit, digit_sebelumnya, digit_sekarang]
    # Posisi 0: Ribuan (tidak ada transisi, hanya frekuensi)
    # Posisi 1: Ratusan (transisi dari Ribuan)
    # Posisi 2: Puluhan (transisi dari Ratusan)
    # Posisi 3: Satuan (transisi dari Puluhan)
    
    # Frekuensi digit pertama (Ribuan)
    freq_ribuan = np.zeros(10)
    # Matriks transisi untuk posisi lainnya
    transitions = np.zeros((3, 10, 10))

    for row in digits:
        freq_ribuan[row[0]] += 1
        transitions[0, row[0], row[1]] += 1  # Ribuan -> Ratusan
        transitions[1, row[1], row[2]] += 1  # Ratusan -> Puluhan
        transitions[2, row[2], row[3]] += 1  # Puluhan -> Satuan

    # Normalisasi untuk mendapatkan probabilitas
    # Tambahkan 1e-6 untuk menghindari pembagian dengan nol
    prob_ribuan = freq_ribuan / (freq_ribuan.sum() + 1e-6)
    
    prob_trans = np.zeros_like(transitions)
    for i in range(3):
        row_sums = transitions[i].sum(axis=1, keepdims=True)
        prob_trans[i] = transitions[i] / (row_sums + 1e-6)

    # Prediksi
    # Untuk setiap posisi, kita ambil probabilitas rata-rata kemunculan setiap digit
    # Ribuan: Berdasarkan frekuensi historis
    # Posisi lain: Probabilitas rata-rata dari semua kemungkinan transisi
    avg_probs = np.zeros((4, 10))
    avg_probs[0] = prob_ribuan
    avg_probs[1] = prob_trans[0].mean(axis=0)
    avg_probs[2] = prob_trans[1].mean(axis=0)
    avg_probs[3] = prob_trans[2].mean(axis=0)
    
    # Ambil top N digit untuk setiap posisi
    result = [np.argsort(probs)[-top_n:][::-1] for probs in avg_probs]
    
    # Kembalikan juga matriks probabilitas untuk kegunaan lain (misal: ensemble)
    return result, avg_probs


def predict_markov_order2(df, top_n=6):
    """
    Prediksi berdasarkan transisi dari dua digit sebelumnya (Order-2).
    Contoh: P(Puluhan | Ribuan, Ratusan), P(Satuan | Ratusan, Puluhan).
    """
    digits = _get_digits(df)
    if digits.shape[0] < 2:
        return None

    # Frekuensi untuk dua posisi pertama
    freq_ribuan = np.zeros(10)
    trans_ratusan = np.zeros((10, 10))
    # Transisi order-2
    trans_puluhan = np.zeros((10, 10, 10)) # (d_ribuan, d_ratusan) -> d_puluhan
    trans_satuan = np.zeros((10, 10, 10))  # (d_ratusan, d_puluhan) -> d_satuan

    for row in digits:
        freq_ribuan[row[0]] += 1
        trans_ratusan[row[0], row[1]] += 1
        trans_puluhan[row[0], row[1], row[2]] += 1
        trans_satuan[row[1], row[2], row[3]] += 1
    
    # Probabilitas rata-rata
    avg_probs = np.zeros((4, 10))
    avg_probs[0] = freq_ribuan / (freq_ribuan.sum() + 1e-6)
    
    sum_ratusan = trans_ratusan.sum(axis=1, keepdims=True)
    avg_probs[1] = (trans_ratusan / (sum_ratusan + 1e-6)).mean(axis=0)

    sum_puluhan = trans_puluhan.sum(axis=2, keepdims=True)
    avg_probs[2] = (trans_puluhan / (sum_puluhan + 1e-6)).mean(axis=(0, 1))
    
    sum_satuan = trans_satuan.sum(axis=2, keepdims=True)
    avg_probs[3] = (trans_satuan / (sum_satuan + 1e-6)).mean(axis=(0, 1))

    result = [np.argsort(probs)[-top_n:][::-1] for probs in avg_probs]
    return result


def predict_markov_hybrid(df, top_n=6):
    """
    Menggabungkan hasil dari Markov Order-1 dan Order-2.
    """
    digits = _get_digits(df)
    if digits.shape[0] < 2:
        return None

    _, probs_o1 = predict_markov(df, top_n=10) # Ambil semua probabilitas
    
    # Dapatkan probabilitas dari Order-2 (kode disederhanakan dari predict_markov_order2)
    freq_ribuan = np.zeros(10)
    trans_ratusan = np.zeros((10, 10))
    trans_puluhan = np.zeros((10, 10, 10))
    trans_satuan = np.zeros((10, 10, 10))
    for row in digits:
        freq_ribuan[row[0]] += 1
        trans_ratusan[row[0], row[1]] += 1
        trans_puluhan[row[0], row[1], row[2]] += 1
        trans_satuan[row[1], row[2], row[3]] += 1
    
    probs_o2 = np.zeros((4, 10))
    probs_o2[0] = freq_ribuan / (freq_ribuan.sum() + 1e-6)
    probs_o2[1] = (trans_ratusan / (trans_ratusan.sum(axis=1, keepdims=True) + 1e-6)).mean(axis=0)
    probs_o2[2] = (trans_puluhan / (trans_puluhan.sum(axis=2, keepdims=True) + 1e-6)).mean(axis=(0, 1))
    probs_o2[3] = (trans_satuan / (trans_satuan.sum(axis=2, keepdims=True) + 1e-6)).mean(axis=(0, 1))

    # Gabungkan probabilitas (misalnya dengan rata-rata)
    hybrid_probs = (probs_o1 + probs_o2) / 2.0
    
    result = [np.argsort(probs)[-top_n:][::-1] for probs in hybrid_probs]
    return result
