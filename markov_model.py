# Simpan sebagai markov_model.py

import pandas as pd
from collections import defaultdict

# Helper function untuk memecah angka menjadi digit
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

# MODIFIKASI UTAMA: Semua fungsi diubah untuk menghasilkan n=7
def top7_markov(df, n=7):
    digits = _prepare_data(df)
    if digits is None or len(digits) < 2:
        return None, None

    predictions = []
    transitions_all = []
    for i in range(1, 5):
        col = f'd{i}'
        transitions = defaultdict(lambda: defaultdict(int))
        for j in range(len(digits) - 1):
            current_digit = digits[col].iloc[j]
            next_digit = digits[col].iloc[j + 1]
            transitions[current_digit][next_digit] += 1
        
        last_digit = digits[col].iloc[-1]
        if last_digit in transitions:
            next_digit_counts = transitions[last_digit]
            sorted_predictions = sorted(next_digit_counts.keys(), key=lambda x: next_digit_counts[x], reverse=True)
            predictions.append(sorted_predictions[:n])
        else:
            # Fallback: prediksi digit paling umum jika state terakhir tidak pernah muncul
            all_counts = digits[col].value_counts()
            predictions.append(all_counts.index.tolist()[:n])
        transitions_all.append(transitions)

    return predictions, transitions_all

def top7_markov_order2(df, n=7):
    digits = _prepare_data(df)
    if digits is None or len(digits) < 3:
        return None

    predictions = []
    for i in range(1, 5):
        col = f'd{i}'
        transitions = defaultdict(lambda: defaultdict(int))
        for j in range(len(digits) - 2):
            current_pair = (digits[col].iloc[j], digits[col].iloc[j + 1])
            next_digit = digits[col].iloc[j + 2]
            transitions[current_pair][next_digit] += 1
            
        last_pair = (digits[col].iloc[-2], digits[col].iloc[-1])
        if last_pair in transitions:
            next_digit_counts = transitions[last_pair]
            sorted_predictions = sorted(next_digit_counts.keys(), key=lambda x: next_digit_counts[x], reverse=True)
            predictions.append(sorted_predictions[:n])
        else:
            # Fallback jika state terakhir tidak pernah muncul
            predictions.append([]) # Return list kosong untuk di-handle oleh hybrid
    
    return predictions

def top7_markov_hybrid(df, n=7):
    # Coba prediksi dengan order-2
    pred_order2 = top7_markov_order2(df, n)
    if pred_order2 is None:
        return top7_markov(df, n)[0]

    # Dapatkan prediksi order-1 sebagai fallback
    pred_order1, _ = top7_markov(df, n)
    if pred_order1 is None:
        return None # Tidak ada data yang cukup bahkan untuk order-1
        
    final_predictions = []
    for i in range(4):
        # Jika prediksi order-2 gagal (list kosong), gunakan order-1
        if not pred_order2[i]:
            final_predictions.append(pred_order1[i])
        else:
            # Padukan hasil jika perlu, atau cukup gunakan order-2
            combined = list(dict.fromkeys(pred_order2[i] + pred_order1[i]))
            final_predictions.append(combined[:n])
            
    return final_predictions
