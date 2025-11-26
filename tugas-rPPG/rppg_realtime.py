"""
nama: Dina Rahma Dita
NIM: 122140184
Tugas rPPG
"""

import cv2
import numpy as np
import time
from collections import deque
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# MediaPipe dengan fallback ke OpenCV
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    import types
    MEDIAPIPE_AVAILABLE = False

    class _DummyFaceDetection:
        def __init__(self, min_detection_confidence=0.7, model_selection=0):
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        def process(self, rgb_frame):
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            detections = []
            h, w = gray.shape
            for (x, y, fw, fh) in faces:
                bbox = types.SimpleNamespace(xmin=x/w, ymin=y/h, width=fw/w, height=fh/h)
                loc = types.SimpleNamespace(relative_bounding_box=bbox)
                detections.append(types.SimpleNamespace(location_data=loc))
            return types.SimpleNamespace(detections=detections)

    mp = types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            face_detection=types.SimpleNamespace(FaceDetection=_DummyFaceDetection)
        )
    )


class rPPGDetector:
    def __init__(self, window_size=300, fps=30):
        self.window_size = window_size
        self.fps = fps
        self.actual_fps = fps

        # Buffer untuk sinyal
        self.signal_buffer = deque(maxlen=window_size)
        self.bpm_history = deque(maxlen=5)
        self.fps_timestamps = deque(maxlen=30)
        self.last_bpm = None

        # Frame tracking untuk monitoring
        self.frame_count = 0
        self.last_print_time = 0
        self.print_interval = 1.0  
        self.print_mode = 'newline'  

        # Deteksi wajah
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.7, model_selection=0
        )

        # Konfigurasi filter (range 50-150 BPM)
        self.lowcut = 0.83
        self.highcut = 2.5
        self.max_bpm_change = 15

        print(f"rPPG Detector diinisialisasi")
        print(f"Deteksi wajah: {'MediaPipe' if MEDIAPIPE_AVAILABLE else 'OpenCV Haar Cascade'}")
        print(f"Window: {window_size} frames ({window_size/fps:.1f}s @ {fps} FPS)")
        print(f"Range BPM: {self.lowcut*60:.0f}-{self.highcut*60:.0f}")

    def detect_face_roi(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if not results.detections:
            return None, None

        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        width = int(bboxC.width * w)
        height = int(bboxC.height * h)

        # Ekstrak area dahi
        roi_y = max(0, y + int(height * 0.1))
        roi_height = min(h - roi_y, int(height * 0.3))
        roi_x = max(0, x + int(width * 0.2))
        roi_width = min(w - roi_x, int(width * 0.6))

        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        bbox = (roi_x, roi_y, roi_width, roi_height)

        return roi, bbox

    def skin_segmentation(self, roi):
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def extract_rgb_signal(self, roi):
        if roi is None or roi.size == 0:
            return None

        skin_mask = self.skin_segmentation(roi)

        if cv2.countNonZero(skin_mask) < 100:
            skin_mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255

        b, g, r = cv2.split(roi)
        r_mean = cv2.mean(r, mask=skin_mask)[0]
        g_mean = cv2.mean(g, mask=skin_mask)[0]
        b_mean = cv2.mean(b, mask=skin_mask)[0]

        return (r_mean, g_mean, b_mean)

    def pos_method(self, rgb_signals):
        """Metode Plane-Orthogonal-to-Skin untuk ekstraksi pulse"""
        rgb_signals = np.array(rgb_signals)
        mean_rgb = np.mean(rgb_signals, axis=0)
        normalized = rgb_signals / mean_rgb
        diff = np.diff(normalized, axis=0)
        pulse = diff[:, 1] - (diff[:, 0] + diff[:, 2]) / 2
        return pulse

    def bandpass_filter(self, data):
        if len(data) < 64:
            return data

        nyquist = self.fps / 2
        low = max(0.01, min(self.lowcut / nyquist, 0.99))
        high = max(low + 0.01, min(self.highcut / nyquist, 0.99))

        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def detrend_signal(self, data):
        window = min(len(data) // 4, 30)
        if window < 3:
            return data
        window = window if window % 2 == 1 else window + 1
        trend = signal.savgol_filter(data, window, 2)
        return data - trend

    def estimate_bpm_fft(self, pulse_signal):
        n = len(pulse_signal)

        # Aplikasikan windowing untuk mengurangi spectral leakage
        window = np.hamming(n)
        windowed_signal = pulse_signal * window

        yf = fft(windowed_signal)
        xf = fftfreq(n, 1/self.actual_fps)

        positive_freqs = xf[:n//2]
        power = np.abs(yf[:n//2]) ** 2

        valid_idx = (positive_freqs >= self.lowcut) & (positive_freqs <= self.highcut)
        valid_freqs = positive_freqs[valid_idx]
        valid_power = power[valid_idx]

        if len(valid_power) == 0:
            return 0, positive_freqs, power

        max_idx = np.argmax(valid_power)

        # Interpolasi parabola untuk akurasi sub-bin
        if 0 < max_idx < len(valid_power) - 1:
            alpha = valid_power[max_idx - 1]
            beta = valid_power[max_idx]
            gamma = valid_power[max_idx + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            dominant_freq = valid_freqs[max_idx] + p * (valid_freqs[1] - valid_freqs[0])
        else:
            dominant_freq = valid_freqs[max_idx]

        return dominant_freq * 60, positive_freqs, power

    def estimate_bpm_peaks(self, pulse_signal):
        min_distance = int(self.actual_fps * 0.5)
        peaks, _ = signal.find_peaks(
            pulse_signal,
            distance=min_distance,
            prominence=np.std(pulse_signal) * 0.3
        )

        if len(peaks) < 2:
            return 0

        peak_intervals = np.diff(peaks)
        median_interval = np.median(peak_intervals)
        return (self.actual_fps / median_interval) * 60

    def update_fps(self, timestamp):
        self.fps_timestamps.append(timestamp)
        if len(self.fps_timestamps) >= 2:
            time_diff = self.fps_timestamps[-1] - self.fps_timestamps[0]
            if time_diff > 0:
                self.actual_fps = (len(self.fps_timestamps) - 1) / time_diff
                self.fps = self.actual_fps

    def validate_bpm(self, new_bpm):
        if new_bpm < 45 or new_bpm > 160:
            return 0

        if self.last_bpm is None:
            self.last_bpm = new_bpm
            return new_bpm

        bpm_change = abs(new_bpm - self.last_bpm)

        if bpm_change > self.max_bpm_change:
            validated = self.last_bpm * 0.7 + new_bpm * 0.3
        else:
            validated = self.last_bpm * 0.4 + new_bpm * 0.6

        self.last_bpm = validated
        return validated

    def process_frame(self, frame, timestamp):
        self.update_fps(timestamp)
        self.frame_count += 1

        roi, bbox = self.detect_face_roi(frame)

        if roi is None:
            cv2.putText(frame, "Wajah tidak terdeteksi", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, 0

        rgb_values = self.extract_rgb_signal(roi)
        if rgb_values is None:
            return frame, 0

        self.signal_buffer.append(rgb_values)

        # Gambar ROI
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "ROI", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        bpm = 0
        bpm_fft = 0
        bpm_peaks = 0

        if len(self.signal_buffer) >= int(self.window_size * 0.6):
            rgb_array = np.array(list(self.signal_buffer))
            pulse_signal = self.pos_method(rgb_array)
            pulse_signal = self.detrend_signal(pulse_signal)
            pulse_signal = self.bandpass_filter(pulse_signal)

            bpm_fft, _, _ = self.estimate_bpm_fft(pulse_signal)
            bpm_peaks = self.estimate_bpm_peaks(pulse_signal)

            if 50 < bpm_fft < 150 and 50 < bpm_peaks < 150:
                bpm = bpm_fft * 0.6 + bpm_peaks * 0.4
            elif 50 < bpm_fft < 150:
                bpm = bpm_fft
            elif 50 < bpm_peaks < 150:
                bpm = bpm_peaks

            if bpm > 0:
                bpm = self.validate_bpm(bpm)
                self.bpm_history.append(bpm)
                bpm = np.mean(list(self.bpm_history))

        # Tampilan
        buffer_pct = (len(self.signal_buffer) / self.window_size) * 100
        status_color = (0, 255, 0) if buffer_pct >= 60 else (0, 165, 255)

        cv2.rectangle(frame, (10, 50), (10 + int(buffer_pct * 3), 70), status_color, -1)
        cv2.rectangle(frame, (10, 50), (310, 70), (255, 255, 255), 2)
        cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Info debug
        cv2.putText(frame, f"FPS: {self.actual_fps:.1f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames: {len(self.signal_buffer)}/{self.window_size}", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if bpm_fft > 0 or bpm_peaks > 0:
            cv2.putText(frame, f"FFT: {bpm_fft:.1f} | Peaks: {bpm_peaks:.1f}", (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if bpm > 0:
            color = (0, 255, 0) if 50 < bpm < 100 else (0, 165, 255)
            cv2.putText(frame, f"Detak Jantung: {bpm:.1f} BPM", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "Menghitung...", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Console monitoring (print setiap interval tertentu untuk tidak lag)
        if timestamp - self.last_print_time >= self.print_interval:
            self.print_monitoring_info(bpm)
            self.last_print_time = timestamp

        return frame, bpm

    def print_monitoring_info(self, bpm):
        """Print monitoring info ke console dengan format rapi"""
        buffer_size = len(self.signal_buffer)

        # Hitung range frame yang diproses
        if buffer_size >= self.window_size:
            frame_start = self.frame_count - self.window_size + 1
            frame_end = self.frame_count
        else:
            frame_start = 1
            frame_end = self.frame_count

        # Durasi window dalam detik
        window_duration = buffer_size / self.actual_fps if self.actual_fps > 0 else 0

        # Format output
        bpm_str = f"{bpm:.1f}" if bpm > 0 else "---"

        # Pilih mode print: newline (scroll) atau overwrite (update di tempat)
        if self.print_mode == 'newline':
            # Mode scroll - setiap update baris baru
            print(f"[Frame {frame_start:04d}-{frame_end:04d}] "
                  f"Buffer: {buffer_size:3d}/{self.window_size} | "
                  f"Window: {window_duration:4.1f}s | "
                  f"FPS: {self.actual_fps:5.1f} | "
                  f"BPM: {bpm_str:>6s}", flush=True)
        else:
            # Mode overwrite - update di baris yang sama
            print(f"\r[Frame {frame_start:04d}-{frame_end:04d}] "
                  f"Buffer: {buffer_size:3d}/{self.window_size} | "
                  f"Window: {window_duration:4.1f}s | "
                  f"FPS: {self.actual_fps:5.1f} | "
                  f"BPM: {bpm_str:>6s}", end='', flush=True)

    def get_current_signal(self):
        if len(self.signal_buffer) < self.window_size // 2:
            return None, None, None, None

        rgb_array = np.array(list(self.signal_buffer))
        pulse_signal = self.pos_method(rgb_array)
        pulse_signal = self.detrend_signal(pulse_signal)
        pulse_signal = self.bandpass_filter(pulse_signal)

        _, freqs, power = self.estimate_bpm_fft(pulse_signal)
        time_array = np.arange(len(pulse_signal)) / self.fps

        return time_array, pulse_signal, freqs, power


def run_rppg_realtime(show_plot=True):
    detector = rPPGDetector(window_size=300, fps=30)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam")
        return

    print("\nSistem rPPG Real-time")
    print("Tekan 'q' untuk keluar, 's' untuk screenshot\n")

    if show_plot:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle('Analisis Sinyal rPPG')

    frame_count = 0
    fps_time = time.time()
    fps_value = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_timestamp = time.time()
            processed_frame, bpm = detector.process_frame(frame, current_timestamp)

            frame_count += 1
            if frame_count % 30 == 0:
                fps_value = 30 / (time.time() - fps_time)
                fps_time = time.time()

            cv2.putText(processed_frame, f"Display FPS: {fps_value:.1f}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Deteksi Detak Jantung rPPG', processed_frame)

            if show_plot and frame_count % 10 == 0:
                time_arr, pulse, freqs, power = detector.get_current_signal()

                if pulse is not None:
                    ax1.clear()
                    ax2.clear()

                    ax1.plot(time_arr, pulse, 'b-', linewidth=1)
                    ax1.set_xlabel('Waktu (detik)')
                    ax1.set_ylabel('Amplitudo')
                    ax1.set_title('Sinyal Pulse (Metode POS)')
                    ax1.grid(True, alpha=0.3)

                    freq_bpm = freqs * 60
                    valid_idx = (freq_bpm >= 40) & (freq_bpm <= 200)
                    ax2.plot(freq_bpm[valid_idx], power[valid_idx], 'r-', linewidth=1)
                    ax2.set_xlabel('Detak Jantung (BPM)')
                    ax2.set_ylabel('Power')
                    ax2.set_title('Spektrum Frekuensi')
                    ax2.grid(True, alpha=0.3)

                    if bpm > 0:
                        ax2.axvline(bpm, color='g', linestyle='--',
                                   label=f'Terdeteksi: {bpm:.1f} BPM')
                        ax2.legend()

                    plt.tight_layout()
                    plt.pause(0.001)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'screenshot_{int(time.time())}.png'
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot tersimpan: {filename}")

    except KeyboardInterrupt:
        print("\nDihentikan oleh user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if show_plot:
            plt.close('all')
        print("Program selesai")


if __name__ == "__main__":
    run_rppg_realtime(show_plot=True)
