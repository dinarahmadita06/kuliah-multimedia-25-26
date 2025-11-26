# rPPG Real-time Heart Rate Detection System

Sistem deteksi detak jantung real-time menggunakan Remote Photoplethysmography (rPPG).

## Fitur (v2.0 - BPM Accuracy Fix)
- ✅ Real-time heart rate detection menggunakan webcam
- ✅ Implementasi metode POS (Plane-Orthogonal-to-Skin)
- ✅ Skin segmentation untuk ROI yang lebih presisi
- ✅ Visualisasi real-time (sinyal + spektrum frekuensi)
- ✅ **Actual FPS tracking** untuk akurasi konversi Hz→BPM
- ✅ **Bandpass filter optimized** (50-150 BPM) untuk HR normal
- ✅ **FFT dengan Hamming windowing** untuk reduce spectral leakage
- ✅ **Peak detection improved** dengan prominence filtering
- ✅ **BPM validation** untuk reject outliers & smooth transitions
- ✅ **Weighted method combination** (FFT 60%, Peaks 40%)
- ✅ **Debug information** (Actual FPS, buffer status, method comparison)
- ✅ MediaPipe face detection (dengan fallback OpenCV)

## Requirements
- Python 3.11 (direkomendasikan, sudah setup di virtual environment)
- Webcam
- Pencahayaan yang cukup

## Cara Menjalankan

### Cara 1: Double-click script (MUDAH)
```
Double-click file: run_rpppg.bat
```

### Cara 2: Manual via terminal
```bash
# Aktifkan virtual environment
venv\Scripts\activate

# Jalankan program
python rpppg_realtime.py
```

## Kontrol Program
- **'q'** - Keluar dari program
- **'s'** - Screenshot

## Tampilan Program
1. **Window Video**: Webcam dengan deteksi wajah dan heart rate
   - ROI (green box) di area dahi
   - Heart Rate: BPM final (smoothed)
   - Buffer progress bar
   - **Debug Info (v2.0):**
     - Actual FPS (harus 20-30)
     - Buffer frames count
     - FFT | Peaks comparison
2. **Window Plot**: Grafik sinyal pulse dan spektrum frekuensi (real-time)

## Tips Penggunaan
1. Pastikan wajah terlihat jelas di kamera
2. Pencahayaan yang cukup (tidak terlalu gelap/terang)
3. Usahakan tetap diam selama pengukuran
4. **Tunggu buffer 60-100%** untuk hasil akurat (6-10 detik)
5. **Check Actual FPS** - jika <20, tingkatkan pencahayaan
6. **Monitor FFT vs Peaks** - difference <10 = good signal

## Expected Results (v2.0)
- **Istirahat:** 60-85 BPM (hijau)
- **Aktif:** 85-120 BPM (orange)
- **Akurasi:** ±3-5 BPM vs pulse oximeter
- **Stabilitas:** Smooth transitions, no jumps

## Troubleshooting

### Error GIL (Python 3.13)
Gunakan Python 3.11 atau 3.12. Virtual environment sudah dikonfigurasi dengan Python 3.11.

### MediaPipe tidak terinstall
Virtual environment sudah include MediaPipe. Gunakan `run_rpppg.bat` atau aktifkan venv terlebih dahulu.

## Teknologi
- OpenCV: Video capture & image processing
- MediaPipe: Face detection
- SciPy: Signal processing & FFT
- Matplotlib: Real-time visualization
- NumPy: Numerical computing
