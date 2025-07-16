# app.py

# --- Fungsi Bantuan ---
def calculate_angka_kontrol(probabilities, top_n=7):
    """
    Menghitung Angka Kontrol berdasarkan matriks probabilitas.
    Jumlah digit yang dihasilkan kini sesuai dengan parameter top_n.
    """
    if probabilities is None or probabilities.shape != (4, 10):
        return {}

    total_probs = np.sum(probabilities, axis=0)
    probs_2d = np.sum(probabilities[2:], axis=0)

    # 1. Angka Kontrol (AK) -> Jumlah digit mengikuti top_n
    ak_global = np.argsort(total_probs)[-top_n:][::-1].tolist()

    # 2. Top 2D (KEP-EKO) -> Jumlah digit mengikuti top_n
    top_2d = np.argsort(probs_2d)[-top_n:][::-1].tolist()

    # 3. Jagoan Posisi (AS-KOP-KEP-EKO) -> Jumlah digit unik mengikuti top_n
    jagoan_per_posisi = np.argmax(probabilities, axis=1).tolist()
    jagoan_final = list(dict.fromkeys(jagoan_per_posisi))
    
    # Mengisi sisa digit dari `ak_global` hingga mencapai `top_n`
    for digit in ak_global:
        if len(jagoan_final) >= top_n:
            break
        if digit not in jagoan_final:
            jagoan_final.append(digit)
            
    # Pengaman jika digit unik masih kurang dari `top_n`
    if len(jagoan_final) < top_n:
        sisa_digit = [d for d in range(10) if d not in jagoan_final]
        needed = top_n - len(jagoan_final)
        jagoan_final.extend(sisa_digit[:needed])

    # 4. Angka Lemah (Hindari) -> Dibiarkan 2 digit karena merupakan angka buangan
    lemah_global = np.argsort(total_probs)[:2].tolist()

    return {
        "Angka Kontrol (AK)": ak_global,
        "Top 2D (KEP-EKO)": top_2d,
        "Jagoan Posisi (AS-KOP-KEP-EKO)": jagoan_final,
        "Angka Lemah (Hindari)": lemah_global,
    }
