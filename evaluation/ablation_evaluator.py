"""
evaluation/ablation_evaluator.py

Evaluator untuk Ablation Study (A-Pyzbar, B-LVLM, C-Hybrid, D-Sistem Lengkap).
Menyimpan & membaca hasil eksperimen, menghitung Accuracy, Precision, Recall,
F1, CER, rata-rata latency & token, serta Database Matching.

Target: 11 gambar x 4 konfigurasi = 44 eksperimen (11 baris per konfigurasi).

Format hasil (ringkas):
  ablation_results.csv  13 kolom utama. Tiap baris = 1 gambar x 1 konfigurasi
                        dan langsung diberi label TP / TN / FP / FN.
  ablation_detail.csv   kolom pendukung (path gambar, latency & token rinci,
                        produk_ditemukan). Terhubung lewat kolom "no".

Aturan label (membandingkan barcode_final dengan barcode asli di kemasan):
  ada barcode & final sama persis        -> TP
  ada barcode & final kosong             -> FN
  ada barcode & final salah angka        -> FP
  tanpa barcode & final kosong           -> TN
  tanpa barcode & final terisi (menebak) -> FP
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
"""

import csv
import math
import os
import re
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "ablation_results.csv")
DETAIL_PATH = os.path.join(BASE_DIR, "ablation_detail.csv")

CONFIGURATIONS = [
    "A-Pyzbar",
    "B-LVLM",
    "C-Hybrid",
    "D-Sistem Lengkap"
]

# File utama (ringkas)
HEADERS = [
    "no", "konfigurasi", "gambar",
    "barcode_asli",   # ground truth dari kemasan fisik
    "ada_barcode",    # Ya / Tidak / kosong (= belum dilabeli)
    "pyzbar", "vision", "final",
    "hasil",          # TP / TN / FP / FN
    "cer", "token", "latency_detik",
    "db_match",       # Ya / Tidak (hanya konfigurasi D)
]

# File detail (tidak perlu dibuka untuk membaca hasil)
DETAIL_HEADERS = [
    "no", "image_path", "produk_ditemukan",
    "latency_preprocessing", "latency_pyzbar", "latency_crop",
    "latency_vision", "latency_database", "latency_llm",
    "token_vision", "token_llm",
]

# Header format lama (20 kolom) -- dikenali untuk konversi otomatis
HEADERS_LAMA_CIRI = ["konfigurasi", "image_path"]

PLACEHOLDER_KOSONG = ("none", "null", "nan", "-")


# Helper
def safe_float(value, default=None):
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)

    text = str(value).strip()

    if text == "" or text.lower() in ["none", "null", "nan", "-"]:
        return default

    try:
        return float(text)
    except (ValueError, TypeError):
        return default


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_barcode(value):
    """Teks barcode; placeholder seperti None / - dianggap kosong."""
    text = normalize_text(value)
    return "" if text.lower() in PLACEHOLDER_KOSONG else text


def normalize_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return None

    text = str(value).strip().lower()

    if text in ["true", "1", "yes", "ya", "benar"]:
        return True

    if text in ["false", "0", "no", "tidak", "salah"]:
        return False

    return None


def ya_tidak(value):
    """True/False/None -> 'Ya'/'Tidak'/''."""
    if value is True:
        return "Ya"
    if value is False:
        return "Tidak"
    return ""


def nama_file(path):
    """Nama file saja; aman untuk path Windows (\\) maupun Linux (/)."""
    return re.split(r"[\\/]", normalize_text(path))[-1]


def hitung_label(prediksi, barcode_asli, ada_barcode):
    """
    Label satu sampel: 'TP' / 'TN' / 'FP' / 'FN', atau '' bila belum bisa dinilai.
      ada_barcode True : prediksi kosong -> FN; sama dengan asli -> TP; beda -> FP
      ada_barcode False: prediksi kosong -> TN; terisi (menebak) -> FP
      ada_barcode None : belum dilabeli -> ''
    """
    if ada_barcode is None:
        return ""

    pred = normalize_barcode(prediksi)
    asli = normalize_barcode(barcode_asli)

    if ada_barcode:
        if not pred:
            return "FN"
        if not asli:
            return ""  # ada barcode, tapi angka aslinya belum diisi
        return "TP" if pred == asli else "FP"

    return "FP" if pred else "TN"


def _kosong_jika_none(value):
    return "" if value is None else value


class AblationEvaluator:

    def __init__(self):
        self._ensure_file()

    # ------------------------------------------------------------------
    # Manajemen file
    # ------------------------------------------------------------------

    @staticmethod
    def _tulis_csv(path, headers, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    @staticmethod
    def _tambah_baris(path, headers, row):
        perlu_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if perlu_header:
                writer.writerow(headers)
            writer.writerow([_kosong_jika_none(row.get(h)) for h in headers])

    @staticmethod
    def _baca_csv(path):
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            return [row for row in csv.DictReader(f) if row]

    @staticmethod
    def _baca_header(path):
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as f:
                return next(csv.reader(f), [])
        except Exception:
            return None

    @staticmethod
    def _cadangkan():
        """Salin ablation_results.csv lama ke file cadangan bertimestamp."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tujuan = os.path.join(BASE_DIR, f"ablation_results_cadangan_{stamp}.csv")
        shutil.copyfile(RESULT_PATH, tujuan)
        return tujuan

    def _ensure_file(self):
        """
        Pastikan kedua file ada dan berformat ringkas.
        Format lama (20 kolom) otomatis dikonversi, data lama dicadangkan dulu.
        """
        os.makedirs(BASE_DIR, exist_ok=True)

        try:
            kosong = not os.path.exists(RESULT_PATH) or os.path.getsize(RESULT_PATH) == 0
        except OSError:
            kosong = True

        if kosong:
            self.reset()
            return

        header = self._baca_header(RESULT_PATH)

        if header == HEADERS:
            if not os.path.exists(DETAIL_PATH) or os.path.getsize(DETAIL_PATH) == 0:
                self._tulis_csv(DETAIL_PATH, DETAIL_HEADERS, [])
            return

        if header and header[:2] == HEADERS_LAMA_CIRI and "barcode_final" in header:
            self._migrasi_dari_format_lama()
            return

        print("[AblationEvaluator] PERINGATAN: header ablation_results.csv tidak dikenali. "
              "File dicadangkan dan hasil baru dibuat.")
        try:
            self._cadangkan()
        except OSError:
            pass
        self.reset()

    def _migrasi_dari_format_lama(self):
        """Konversi ablation_results.csv format lama (20 kolom) ke format ringkas + detail."""
        cadangan = self._cadangkan()

        lama = [
            r for r in self._baca_csv(RESULT_PATH)
            if normalize_text(r.get("konfigurasi")) in CONFIGURATIONS
        ]

        utama, detail, tanpa_barcode = [], [], []

        for no, r in enumerate(lama, start=1):
            baris_utama, baris_detail = self._susun_baris(no, r)
            utama.append([baris_utama[h] for h in HEADERS])
            detail.append([_kosong_jika_none(baris_detail.get(h)) for h in DETAIL_HEADERS])

            if baris_utama["ada_barcode"] == "Tidak" and baris_utama["gambar"] not in tanpa_barcode:
                tanpa_barcode.append(baris_utama["gambar"])

        self._tulis_csv(RESULT_PATH, HEADERS, utama)
        self._tulis_csv(DETAIL_PATH, DETAIL_HEADERS, detail)

        print(
            f"[AblationEvaluator] ablation_results.csv dikonversi ke format ringkas "
            f"({len(utama)} baris). Cadangan format lama: {os.path.basename(cadangan)}"
        )
        if tanpa_barcode:
            print(
                "[AblationEvaluator] ada_barcode diisi 'Tidak' otomatis (barcode_asli kosong) untuk gambar: "
                + ", ".join(tanpa_barcode)
                + ". Pastikan gambar-gambar itu memang tanpa barcode; jika tidak, isi barcode_asli / "
                  "ubah ada_barcode di CSV."
            )

    def reset(self):
        self._tulis_csv(RESULT_PATH, HEADERS, [])
        self._tulis_csv(DETAIL_PATH, DETAIL_HEADERS, [])

        print(f"[INFO] Ablation result di-reset: {RESULT_PATH}")

    # ------------------------------------------------------------------
    # Simpan & baca
    # ------------------------------------------------------------------

    @staticmethod
    def _susun_baris(no, r):
        """
        Ubah satu hasil eksperimen (kunci seperti keluaran process_ablation:
        barcode_pyzbar, barcode_vision, barcode_final, latency_total, token_total_ai,
        database_match, ...) menjadi satu baris file utama + satu baris file detail.
        """
        asli = normalize_barcode(r.get("barcode_asli"))
        pyzbar = normalize_barcode(r.get("barcode_pyzbar"))
        vision = normalize_barcode(r.get("barcode_vision"))
        final = normalize_barcode(r.get("barcode_final"))

        # ada_barcode: bisa dikirim eksplisit; kalau tidak, barcode_asli terisi = ada.
        ada = normalize_bool(r.get("ada_barcode"))
        if ada is None:
            ada = bool(asli)

        utama = {
            "no": no,
            "konfigurasi": normalize_text(r.get("konfigurasi")),
            "gambar": nama_file(r.get("image_path")),
            "barcode_asli": asli,
            "ada_barcode": ya_tidak(ada),
            "pyzbar": pyzbar,
            "vision": vision,
            "final": final,
            "hasil": hitung_label(final, asli, ada),
            "cer": _kosong_jika_none(r.get("cer")),
            "token": _kosong_jika_none(r.get("token_total_ai")),
            "latency_detik": _kosong_jika_none(r.get("latency_total")),
            "db_match": ya_tidak(normalize_bool(r.get("database_match"))),
        }

        detail = {"no": no, "image_path": normalize_text(r.get("image_path"))}
        for kolom in DETAIL_HEADERS:
            if kolom not in detail:
                detail[kolom] = _kosong_jika_none(r.get(kolom))

        return utama, detail

    def _nomor_berikutnya(self):
        nomor = [
            int(r["no"]) for r in self._baca_csv(RESULT_PATH)
            if normalize_text(r.get("no")).isdigit()
        ]
        return (max(nomor) if nomor else 0) + 1

    def save_result(self, result):
        if not isinstance(result, dict):
            raise TypeError("result harus berupa dictionary.")

        konfigurasi = normalize_text(result.get("konfigurasi"))

        if konfigurasi not in CONFIGURATIONS:
            raise ValueError(
                f"Konfigurasi tidak valid: {konfigurasi}\n"
                f"Konfigurasi yang diperbolehkan: {CONFIGURATIONS}"
            )

        utama, detail = self._susun_baris(self._nomor_berikutnya(), result)

        self._tambah_baris(RESULT_PATH, HEADERS, utama)
        self._tambah_baris(DETAIL_PATH, DETAIL_HEADERS, detail)

    def load_rows(self):
        """Baris file utama digabung dengan file detail (lewat kolom 'no')."""
        detail = {normalize_text(r.get("no")): r for r in self._baca_csv(DETAIL_PATH)}

        gabungan = []
        for r in self._baca_csv(RESULT_PATH):
            baris = dict(detail.get(normalize_text(r.get("no")), {}))
            baris.update(r)
            gabungan.append(baris)

        return gabungan

    def perbarui_label_csv(self):
        """
        Hitung ulang kolom 'hasil' di ablation_results.csv dari kolom mentah
        (barcode_asli, ada_barcode, final). Berguna setelah ada_barcode /
        barcode_asli diedit manual di Excel.
        """
        rows = self._baca_csv(RESULT_PATH)
        if not rows:
            return False

        for r in rows:
            r["hasil"] = hitung_label(
                r.get("final"), r.get("barcode_asli"), normalize_bool(r.get("ada_barcode"))
            )

        try:
            self._tulis_csv(RESULT_PATH, HEADERS, [[r.get(h) or "" for h in HEADERS] for r in rows])
        except PermissionError:
            print("[AblationEvaluator] ablation_results.csv sedang dibuka (mis. di Excel), "
                  "kolom 'hasil' tidak diperbarui. Perhitungan ringkasan tetap memakai data terbaru.")
            return False
        return True

    def count_per_configuration(self):
        rows = self.load_rows()
        result = {config: 0 for config in CONFIGURATIONS}

        for row in rows:
            config = normalize_text(row.get("konfigurasi"))
            if config in result:
                result[config] += 1

        return result

    # ------------------------------------------------------------------
    # Metrik
    # ------------------------------------------------------------------

    @staticmethod
    def confusion(rows):
        """
        Confusion matrix per konfigurasi, dihitung ulang dari kolom mentah
        (barcode_asli, ada_barcode, final) sehingga tetap benar walau CSV diedit.

        Sampel positif (ada barcode):
          - final kosong          -> FN
          - final == barcode_asli -> TP
          - final salah           -> FP

        Sampel negatif (tanpa barcode):
          - final kosong          -> TN
          - final terisi          -> FP (menebak barcode yang tidak ada)

        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Baris yang ada_barcode-nya kosong (belum dilabeli) tidak dihitung.
        """
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for row in rows:
            asli = normalize_barcode(row.get("barcode_asli"))
            pred = row.get("final") if "final" in row else row.get("barcode_final")

            if "ada_barcode" in row:
                ada = normalize_bool(row.get("ada_barcode"))
            else:
                ada = bool(asli)  # baris format lama

            label = hitung_label(pred, asli, ada)

            if label == "TP":
                tp += 1
            elif label == "TN":
                tn += 1
            elif label == "FP":
                fp += 1
            elif label == "FN":
                fn += 1

        valid_rows = tp + tn + fp + fn

        accuracy = (tp + tn) / valid_rows * 100 if valid_rows > 0 else None
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else None
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else None

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        return {
            "n": valid_rows,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    @staticmethod
    def average(rows, column):
        values = []

        for row in rows:
            value = safe_float(row.get(column))
            if value is None:
                continue
            values.append(value)

        if not values:
            return None

        return sum(values) / len(values)

    @staticmethod
    def database_match_percentage(rows):
        values = []

        for row in rows:
            value = normalize_bool(row.get("db_match"))
            if value is None:
                continue
            values.append(value)

        if not values:
            return None

        benar = sum(1 for value in values if value)

        return benar / len(values) * 100

    @staticmethod
    def average_cer(rows):
        values = []

        for row in rows:
            value = safe_float(row.get("cer"))
            if value is None:
                continue
            values.append(value)

        if not values:
            return None

        return sum(values) / len(values)

    def compute_summary(self):
        rows = self.load_rows()

        if not rows:
            print("[INFO] Belum ada hasil Ablation Study.")
            return None

        hasil = {}

        for konfigurasi in CONFIGURATIONS:
            data = [
                row for row in rows
                if normalize_text(row.get("konfigurasi")) == konfigurasi
            ]

            if not data:
                continue

            metrics = self.confusion(data)
            rata_cer = self.average_cer(data)

            latency_total = self.average(data, "latency_detik")
            latency_preprocessing = self.average(data, "latency_preprocessing")
            latency_pyzbar = self.average(data, "latency_pyzbar")
            latency_crop = self.average(data, "latency_crop")
            latency_vision = self.average(data, "latency_vision")
            latency_database = self.average(data, "latency_database")
            latency_llm = self.average(data, "latency_llm")

            token_vision = self.average(data, "token_vision")
            token_llm = self.average(data, "token_llm")
            token_total = self.average(data, "token")

            database_matching = self.database_match_percentage(data)

            hasil[konfigurasi] = {
                "n": len(data),
                "valid_n": metrics["n"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "cer": rata_cer,
                "latency": latency_total,
                "latency_preprocessing": latency_preprocessing,
                "latency_pyzbar": latency_pyzbar,
                "latency_crop": latency_crop,
                "latency_vision": latency_vision,
                "latency_database": latency_database,
                "latency_llm": latency_llm,
                "token": token_total,
                "token_vision": token_vision,
                "token_llm": token_llm,
                "database_matching": database_matching
            }

        return hasil

    # ------------------------------------------------------------------
    # Cetak ke terminal
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_pct(value):
        if value is None:
            return "-"
        return f"{value:.1f}%"

    @staticmethod
    def fmt_num(value):
        if value is None:
            return "-"
        return f"{value:.3f}"

    @staticmethod
    def fmt_detik(value):
        if value is None:
            return "-"
        if value < 1:
            return f"{value * 1000:.2f} ms"
        return f"{value:.2f} detik"

    def print_summary(self):
        # Segarkan kolom 'hasil' dulu, supaya sesuai bila CSV diedit manual.
        self.perbarui_label_csv()

        hasil = self.compute_summary()

        if not hasil:
            return None

        print("[ABLATION STUDY] Perbandingan Konfigurasi")

        headers = [
            "Konfigurasi", "n", "Accuracy", "Precision",
            "Recall", "F1", "CER", "Latency", "Token", "DB Match"
        ]

        rows = []

        for konfigurasi in CONFIGURATIONS:
            if konfigurasi not in hasil:
                continue

            h = hasil[konfigurasi]

            rows.append([
                konfigurasi,
                h["n"],
                self.fmt_pct(h["accuracy"]),
                self.fmt_pct(h["precision"]),
                self.fmt_pct(h["recall"]),
                self.fmt_pct(h["f1"]),
                self.fmt_num(h["cer"]),
                self.fmt_detik(h["latency"]),
                f"{h['token']:.1f}" if h["token"] is not None else "-",
                self.fmt_pct(h["database_matching"])
            ])

        # Hitung lebar kolom
        widths = [len(header) for header in headers]

        for row in rows:
            for i, value in enumerate(row):
                widths[i] = max(widths[i], len(str(value)))

        print("  " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
        print("  " + "-+-".join("-" * widths[i] for i in range(len(widths))))

        for row in rows:
            print("  " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))

        # Confusion matrix
        print("[DETAIL CONFUSION MATRIX]")
        print("  Accuracy = (TP + TN) / (TP + TN + FP + FN)")
        print("  TN = gambar tanpa barcode (ada_barcode = Tidak) & final kosong")

        cm_headers = ["Konfigurasi", "TP", "TN", "FP", "FN", "Total"]
        cm_rows = []

        for konfigurasi in CONFIGURATIONS:
            if konfigurasi not in hasil:
                continue

            h = hasil[konfigurasi]
            cm_rows.append([
                konfigurasi, h["tp"], h["tn"], h["fp"], h["fn"],
                h["tp"] + h["tn"] + h["fp"] + h["fn"],
            ])

        cm_widths = [len(x) for x in cm_headers]
        for r in cm_rows:
            for i, v in enumerate(r):
                cm_widths[i] = max(cm_widths[i], len(str(v)))

        print("  " + " | ".join(cm_headers[i].ljust(cm_widths[i]) for i in range(len(cm_headers))))
        print("  " + "-+-".join("-" * w for w in cm_widths))
        for r in cm_rows:
            print("  " + " | ".join(str(r[i]).ljust(cm_widths[i]) for i in range(len(r))))

        # Detail latency
        print("[DETAIL LATENCY]")

        for konfigurasi in CONFIGURATIONS:
            if konfigurasi not in hasil:
                continue

            h = hasil[konfigurasi]

            print(f"{konfigurasi}:")
            print("  Preprocessing : " + self.fmt_detik(h["latency_preprocessing"]))
            print("  Pyzbar        : " + self.fmt_detik(h["latency_pyzbar"]))
            print("  Cropping      : " + self.fmt_detik(h["latency_crop"]))
            print("  Vision AI     : " + self.fmt_detik(h["latency_vision"]))
            print("  Database      : " + self.fmt_detik(h["latency_database"]))
            print("  AI Explainer  : " + self.fmt_detik(h["latency_llm"]))
            print("  TOTAL         : " + self.fmt_detik(h["latency"]))

        # Detail token
        print("[DETAIL TOKEN]")

        for konfigurasi in CONFIGURATIONS:
            if konfigurasi not in hasil:
                continue

            h = hasil[konfigurasi]

            print(f"{konfigurasi}:")
            print("  Vision AI    : " + (f"{h['token_vision']:.1f}" if h["token_vision"] is not None else "0.0"))
            print("  AI Explainer : " + (f"{h['token_llm']:.1f}" if h["token_llm"] is not None else "0.0"))
            print("  TOTAL        : " + (f"{h['token']:.1f}" if h["token"] is not None else "0.0"))

        # Validasi jumlah data
        print("[VALIDASI JUMLAH DATA]")

        total_rows = sum(h["n"] for h in hasil.values())

        for konfigurasi in CONFIGURATIONS:
            if konfigurasi in hasil:
                print(f"  {konfigurasi:<20} : {hasil[konfigurasi]['n']} sampel")
            else:
                print(f"  {konfigurasi:<20} : 0 sampel")

        print(f"  TOTAL BARIS ABLATION : {total_rows}")

        expected = 11 * len(CONFIGURATIONS)
        print(f"  TARGET EKSPERIMEN    : {expected}")

        if total_rows == expected:
            print("  [OK] Seluruh eksperimen Ablation sudah tersimpan.")

        return hasil