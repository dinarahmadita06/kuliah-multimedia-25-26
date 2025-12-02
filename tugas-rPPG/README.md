# rPPG Realtime – Deteksi Detak Jantung dari Webcam

Sistem deteksi detak jantung real-time menggunakan Remote Photoplethysmography (rPPG).

## Fitur Utama (sesuai kode)
- **Deteksi wajah**: MediaPipe FaceDetection; otomatis fallback ke OpenCV Haar Cascade jika MediaPipe tidak tersedia.
- **ROI dahi (jidat)**: Di bagian atas bounding box wajah dengan geometri tetap di kode:
   - tinggi ≈ 20% dari tinggi wajah
   - lebar ≈ 40% dari lebar wajah
   - posisi X digeser ≈ 30% ke dalam (lebih ke tengah)
- **Segmentasi kulit (YCrCb)**: Threshold warna kulit + morfologi (opening & closing) untuk menstabilkan rata-rata RGB ROI.
- **Ekstraksi sinyal (POS)**: Gunakan metode POS pada RGB untuk memperoleh sinyal pulse.
- **Detrend + Bandpass**:
   - Detrending dengan Savitzky–Golay
   - Bandpass Butterworth orde 4, 0.83–2.5 Hz (≈50–150 BPM)
- **Estimasi BPM (hybrid)**:
   - FFT (dengan window Hamming dan interpolasi parabola) → kandidat BPM
   - Deteksi puncak di domain waktu → kandidat BPM
   - Penggabungan: 60% FFT + 40% Peaks (jika keduanya valid); jika hanya satu valid, gunakan yang valid
- **Validasi & smoothing**:
   - Tolak di luar 45–160 BPM
   - Batasi perubahan maksimal ±15 BPM
   - Simpan riwayat pendek (deque) dan rata-ratakan untuk hasil halus
- **Rolling log & Plot realtime (Jupyter)**:
   - Dua area output: log (append-only) dan plot (dibersihkan setiap update)
   - Plot diperbarui setiap `update_interval` frame (default 10) menggunakan `clear_output(wait=True)`
- **Snapshot untuk export**:
   - Simpan otomatis plot terbaru ke `rppg_plot_latest.png`
   - Tombol `s` menyimpan `screenshot_*.png` dari jendela OpenCV

## Parameter Penting
- `window_size`: panjang buffer sliding (default 300 frame).
- `fps`: target frame rate (default 30; sistem mengukur FPS aktual).
- `lowcut/highcut`: 0.83–2.5 Hz (≈50–150 BPM).
- `update_interval`: frekuensi pembaruan plot (mis. 10 frame).

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

## Teknologi
- OpenCV: Video capture & image processing
- MediaPipe: Face detection
- SciPy: Signal processing & FFT
- Matplotlib: Real-time visualization
- NumPy: Numerical computing
