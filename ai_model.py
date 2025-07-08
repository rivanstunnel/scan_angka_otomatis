# Ganti fungsi lama di ai_model.py dengan yang ini

from collections import defaultdict
# Pastikan Anda mengimpor fungsi-fungsi ini di bagian atas file ai_model.py
from markov_model import top_n_markov_hybrid
# Nama fungsi LSTM mungkin berbeda, sesuaikan jika perlu (misal: top_n_lstm)
from ai_model import top_n_lstm 

def top_n_ensemble(df, lokasi, top_n=7):
    """
    Gabungan prediksi dari Markov Hybrid dan LSTM AI.
    Nama fungsi ini harus cocok dengan yang Anda impor di app.py.
    """
    # 1. Panggil model Markov Hybrid dengan variabel top_n yang benar
    pred_markov = top_n_markov_hybrid(df, top_n=top_n)

    # 2. Panggil model LSTM dengan variabel top_n yang benar
    pred_lstm = top_n_lstm(df, lokasi=lokasi, top_n=top_n)

    # Cek jika salah satu prediksi gagal
    if not pred_markov or not pred_lstm:
        print("Peringatan: Salah satu model (Markov/LSTM) gagal prediksi. Ensemble dibatalkan.")
        return None
    
    ensemble_predictions = []
    for i in range(4):
        # Beri bobot berdasarkan peringkat
        scores = defaultdict(float)
        
        # Bobot untuk LSTM (lebih dipercaya)
        for rank, digit in enumerate(pred_lstm[i]):
            scores[digit] += (top_n - rank) * 1.5 
            
        # Bobot untuk Markov Hybrid
        for rank, digit in enumerate(pred_markov[i]):
            scores[digit] += (top_n - rank) * 1.0

        # Urutkan digit berdasarkan skor gabungan tertinggi
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_n_list = [digit for digit, score in sorted_scores[:top_n]]
        
        # Jika hasil kurang dari top_n, lengkapi dengan angka acak
        if len(top_n_list) < top_n:
            import numpy as np
            remaining = [d for d in range(10) if d not in top_n_list]
            top_n_list.extend(np.random.choice(remaining, size=top_n-len(top_n_list), replace=False))

        ensemble_predictions.append(top_n_list)
        
    return ensemble_predictions
