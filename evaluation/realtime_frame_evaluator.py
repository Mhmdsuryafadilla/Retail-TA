"""
evaluation/realtime_frame_evaluator.py

Pencatatan evaluasi pemindaian barcode secara realtime.

Data yang dicatat:
- scan_id
- start_time
- elapsed_time
- frame_number
- barcode_detected
- detection_frame
- detection_time
- processed_frame
- processed_time
- fps_actual
- frames_checked
- total_processing_time

Selain summary per sesi, sistem juga menyimpan data setiap frame
ke realtime_frame_log.csv untuk kebutuhan analisis lebih detail.
"""

import csv
import os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)

SUMMARY_PATH = os.path.join(BASE_DIR, "realtime_scan_summary.csv")
FRAME_LOG_PATH = os.path.join(BASE_DIR, "realtime_frame_log.csv")

SUMMARY_HEADERS = [
    "scan_id",
    "start_time",
    "elapsed_time",
    "frame_number",
    "barcode_detected",
    "detection_frame",
    "detection_time",
    "processed_frame",
    "processed_time",
    "fps_actual",
    "frames_checked",
    "total_processing_time",
]

FRAME_HEADERS = [
    "scan_id",
    "frame_number",
    "elapsed_time",
    "barcode_detected",
]


class RealtimeFrameEvaluator:

    def __init__(self):
        self._ensure_files()

    def _ensure_files(self):
        if not os.path.exists(SUMMARY_PATH) or os.path.getsize(SUMMARY_PATH) == 0:
            with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(SUMMARY_HEADERS)

        if not os.path.exists(FRAME_LOG_PATH) or os.path.getsize(FRAME_LOG_PATH) == 0:
            with open(FRAME_LOG_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(FRAME_HEADERS)

    def get_next_scan_id(self):
        self._ensure_files()

        try:
            with open(SUMMARY_PATH, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            if not rows:
                return 1

            ids = []
            for row in rows:
                try:
                    ids.append(int(row["scan_id"]))
                except (ValueError, TypeError, KeyError):
                    continue

            if not ids:
                return 1

            return max(ids) + 1

        except Exception:
            return 1

    def save_session(
        self,
        scan_id,
        start_time,
        elapsed_time,
        frame_number,
        barcode_detected,
        detection_frame,
        detection_time,
        processed_frame,
        processed_time,
        fps_actual,
        frames_checked,
        total_processing_time,
        frame_records=None
    ):
        self._ensure_files()

        # Summary per sesi
        with open(SUMMARY_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                scan_id,
                start_time,
                self._safe_round(elapsed_time),
                frame_number,
                bool(barcode_detected),
                detection_frame,
                self._safe_round(detection_time),
                processed_frame,
                self._safe_round(processed_time),
                self._safe_round(fps_actual),
                frames_checked,
                self._safe_round(total_processing_time),
            ])

        # Detail tiap frame
        if frame_records:
            with open(FRAME_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for record in frame_records:
                    writer.writerow([
                        scan_id,
                        record.get("frame_number"),
                        self._safe_round(record.get("elapsed_time")),
                        bool(record.get("barcode_detected", False)),
                    ])

    def read_summaries(self):
        self._ensure_files()

        with open(SUMMARY_PATH, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def calculate_summary(self):
        rows = self.read_summaries()

        if not rows:
            return None

        fps = []
        detection_frame = []
        detection_time = []
        processed_frame = []
        processed_time = []
        frames_checked = []
        processing_time = []

        for row in rows:
            values = [
                ("fps", fps),
                ("detection_frame", detection_frame),
                ("detection_time", detection_time),
                ("processed_frame", processed_frame),
                ("processed_time", processed_time),
                ("frames_checked", frames_checked),
                ("total_processing_time", processing_time),
            ]

            for field, target in values:
                value = row.get(field)
                if value in (None, "", "None"):
                    continue
                try:
                    target.append(float(value))
                except (ValueError, TypeError):
                    pass

        def average(values):
            if not values:
                return None
            return round(sum(values) / len(values), 3)

        return {
            "jumlah_sesi": len(rows),
            "rata_fps_actual": average(fps),
            "rata_detection_frame": average(detection_frame),
            "rata_detection_time": average(detection_time),
            "rata_processed_frame": average(processed_frame),
            "rata_processed_time": average(processed_time),
            "rata_frames_checked": average(frames_checked),
            "rata_total_processing_time": average(processing_time),
        }

    def print_summary(self):
        result = self.calculate_summary()

        print()
        print("-" * 70)
        print("[TAHAP 3B] EVALUASI REALTIME FRAME")
        print("-" * 70)

        if not result:
            print("Belum ada data realtime.")
            print("Lakukan pengujian menggunakan Live Camera terlebih dahulu.")
            return None

        print(f"  Jumlah sesi scan        : {result['jumlah_sesi']}")
        print(f"  Rata-rata FPS aktual   : {self._format(result['rata_fps_actual'])} FPS")
        print(f"  Rata-rata frame deteksi: {self._format(result['rata_detection_frame'])}")
        print(f"  Rata-rata waktu deteksi: {self._format(result['rata_detection_time'])} detik")
        print(f"  Rata-rata frame proses : {self._format(result['rata_processed_frame'])}")
        print(f"  Rata-rata waktu proses : {self._format(result['rata_processed_time'])} detik")
        print(f"  Rata-rata frame diperiksa: {self._format(result['rata_frames_checked'])}")
        print(f"  Rata-rata waktu processing: {self._format(result['rata_total_processing_time'])} detik")
        print("-" * 70)

        return result

    @staticmethod
    def _safe_round(value):
        if value in (None, "", "None"):
            return ""
        try:
            return round(float(value), 3)
        except (ValueError, TypeError):
            return ""

    @staticmethod
    def _format(value):
        if value is None:
            return "-"
        return f"{value:.3f}"