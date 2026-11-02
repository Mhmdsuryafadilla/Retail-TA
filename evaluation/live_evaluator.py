#live_evaluator.py
"""
LiveEvaluator -- log scan + evaluasi batch.

Format log (ringkas):
  evaluation_log.csv     13 kolom utama, mudah dibaca. Tiap baris langsung diberi
                         label TP / TN / FP / FN untuk pyzbar, vision, dan final.
  evaluation_detail.csv  kolom pendukung (kondisi, checksum, retrieval DB, latency
                         rinci, teks konteks & jawaban AI). Terhubung lewat kolom "no".

Aturan label (membandingkan hasil baca dengan barcode asli di kemasan):
  ada barcode & prediksi sama persis        -> TP
  ada barcode & prediksi kosong             -> FN
  ada barcode & prediksi salah angka        -> FP
  tanpa barcode & prediksi kosong           -> TN
  tanpa barcode & prediksi terisi (menebak) -> FP
  Accuracy = (TP + TN) / (TP + TN + FP + FN)

Semua scan dievaluasi, terbaca maupun tidak terbaca (n = jumlah seluruh baris log).
Kolom ada_barcode yang kosong diisi otomatis dari barcode_asli:
terisi -> Ya, kosong -> Tidak. Isi manual di CSV bila kenyataannya berbeda.
"""
import csv
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "evaluation_log.csv")
DETAIL_PATH = os.path.join(BASE_DIR, "evaluation_detail.csv")

# Asumsi harga per 1000 token (Rupiah) -- sesuaikan dengan provider/router yang dipakai
HARGA_PER_1000_TOKEN_DEFAULT = 150

# File utama (ringkas)
LOG_HEADERS = [
    "no", "waktu", "mode",
    "barcode_asli",   # ground truth dari kemasan fisik
    "ada_barcode",    # Ya / Tidak (kosong = otomatis dari barcode_asli)
    "pyzbar", "vision", "final",
    "hasil_pyzbar", "hasil_vision", "hasil_final",   # TP / TN / FP / FN
    "token", "latency_detik",
]

# File detail (tidak perlu dibuka untuk membaca hasil, tapi dipakai laporan tertentu)
DETAIL_HEADERS = [
    "no",
    "produk_ditemukan",
    "kondisi_pencahayaan", "kondisi_fisik_barcode", "image_detail_level",
    "pyzbar_checksum_valid", "vision_checksum_valid", "gambar_di_crop",
    "agreement_pyzbar_vision", "cer_pyzbar", "cer_vision",
    "jumlah_pembanding_ditampilkan", "jumlah_pembanding_valid_di_db",
    "context_recall_estimate_persen", "akurasi_filter_kategori_persen",
    "token_vision_ai", "token_llm_explainer", "estimasi_biaya_rupiah",
    "latency_pyzbar_detik", "latency_vision_detik", "latency_llm_detik",
    "context_text", "ai_answer_text",
]

# Nama metode (untuk laporan) -> kolom prediksi & kolom label di file utama
PRED_KEYS = {
    "pyzbar": "pyzbar",
    "vision_ai": "vision",
    "hybrid": "final",
}
HASIL_KEYS = {
    "pyzbar": "hasil_pyzbar",
    "vision_ai": "hasil_vision",
    "hybrid": "hasil_final",
}

KOLOM_KONDISI = ["kondisi_pencahayaan", "kondisi_fisik_barcode", "image_detail_level"]


def fmt_pct(v):
    return f"{v}%" if v is not None else "-"


def _ya_tidak(v):
    """True/False/None -> 'Ya'/'Tidak'/'-'."""
    if v is None:
        return "-"
    return "Ya" if v else "Tidak"


def _label_ada(v):
    """True/False/None -> 'Ya'/'Tidak'/'' (untuk kolom ada_barcode di CSV)."""
    if v is True:
        return "Ya"
    if v is False:
        return "Tidak"
    return ""


def _tulis_csv(path, headers, rows):
    """Tulis ulang seluruh file: satu header + rows."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _tambah_baris(path, headers, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["" if row.get(h) is None else row.get(h) for h in headers]
        )


def _int_atau_none(value):
    return None if value in (None, "", "None") else int(value)


class LiveEvaluator:

    def __init__(self, harga_per_1000_token=HARGA_PER_1000_TOKEN_DEFAULT):
        self.harga_per_1000_token = harga_per_1000_token
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # Manajemen file log
    # ------------------------------------------------------------------

    @staticmethod
    def _tulis_baru():
        _tulis_csv(LOG_PATH, LOG_HEADERS, [])
        _tulis_csv(DETAIL_PATH, DETAIL_HEADERS, [])

    @staticmethod
    def _cadangkan():
        """Salin evaluation_log.csv lama ke file cadangan bertimestamp."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tujuan = os.path.join(BASE_DIR, f"evaluation_log_cadangan_{stamp}.csv")
        shutil.copyfile(LOG_PATH, tujuan)
        return tujuan

    def _ensure_log_file(self):
        """
        Pastikan kedua file log ada dan berformat ringkas.
        Format lama (34 kolom) otomatis dikonversi, data lama dicadangkan dulu.
        """
        if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
            self._tulis_baru()
            return

        try:
            with open(LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
                header = next(csv.reader(f), [])
        except Exception as e:
            print(f"[LiveEvaluator] Gagal membaca evaluation_log.csv: {e}")
            self._cadangkan()
            self._tulis_baru()
            return

        if header == LOG_HEADERS:
            if not os.path.exists(DETAIL_PATH) or os.path.getsize(DETAIL_PATH) == 0:
                _tulis_csv(DETAIL_PATH, DETAIL_HEADERS, [])
            return

        if "timestamp" in header and "barcode_dipakai" in header:
            self._migrasi_dari_format_lama()
            return

        print("[LiveEvaluator] PERINGATAN: header evaluation_log.csv tidak dikenali. "
              "File dicadangkan dan log baru dibuat.")
        self._cadangkan()
        self._tulis_baru()

    def _migrasi_dari_format_lama(self):
        """Konversi evaluation_log.csv format lama (34 kolom) ke format ringkas + detail."""
        cadangan = self._cadangkan()

        with open(LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
            rows_lama = [
                r for r in csv.DictReader(f)
                if r and (r.get("timestamp") or "").strip().lower() != "timestamp"
            ]

        def v(r, kunci):
            return r.get(kunci) or ""

        utama, detail, dicek = [], [], []

        for no, r in enumerate(rows_lama, start=1):
            asli = self._normalisasi_barcode(r.get("barcode_asli"))
            pyz = self._normalisasi_barcode(r.get("barcode_pyzbar"))
            vis = self._normalisasi_barcode(r.get("barcode_vision_tebakan"))
            fin = self._normalisasi_barcode(r.get("barcode_dipakai"))

            ada = self._parse_bool(r.get("ground_truth_barcode_present"))
            if ada is None:
                ada = bool(asli)
                if not asli:
                    dicek.append(no)

            utama.append([
                no, v(r, "timestamp"), v(r, "mode"), asli, _label_ada(ada),
                pyz, vis, fin,
                self.hitung_label(pyz, asli, ada),
                self.hitung_label(vis, asli, ada),
                self.hitung_label(fin, asli, ada),
                v(r, "token_total"), v(r, "latency_total_detik"),
            ])

            sumber = dict(r)
            sumber["no"] = no
            detail.append([v(sumber, h) for h in DETAIL_HEADERS])

        _tulis_csv(LOG_PATH, LOG_HEADERS, utama)
        _tulis_csv(DETAIL_PATH, DETAIL_HEADERS, detail)

        print(
            f"[LiveEvaluator] evaluation_log.csv dikonversi ke format ringkas "
            f"({len(utama)} baris). Cadangan format lama: {os.path.basename(cadangan)}"
        )
        if dicek:
            print(
                "[LiveEvaluator] PERHATIAN: kolom ada_barcode diisi 'Tidak' otomatis untuk baris no "
                f"{', '.join(map(str, dicek))} karena barcode_asli-nya kosong. Itu BELUM tentu gambar "
                "tanpa barcode (bisa juga produk tidak ditemukan di database). Periksa gambarnya, "
                "lalu ubah ada_barcode / barcode_asli di CSV bila perlu."
            )

    def log_to_csv(self, data):
        """Tambahkan satu baris ke file ringkas dan satu baris ke file detail."""
        self._ensure_log_file()
        no = self._jumlah_baris_utama() + 1
        data["no"] = no

        token = data.get("token_total")
        utama = {
            "no": no,
            "waktu": data.get("timestamp"),
            "mode": data.get("mode"),
            "barcode_asli": self._normalisasi_barcode(data.get("barcode_asli")),
            "ada_barcode": data.get("ada_barcode", ""),
            "pyzbar": self._normalisasi_barcode(data.get("barcode_pyzbar")),
            "vision": self._normalisasi_barcode(data.get("barcode_vision_tebakan")),
            "final": self._normalisasi_barcode(data.get("barcode_dipakai")),
            "hasil_pyzbar": data.get("hasil_pyzbar", ""),
            "hasil_vision": data.get("hasil_vision", ""),
            "hasil_final": data.get("hasil_final", ""),
            "token": token,
            "latency_detik": data.get("latency_total_detik"),
        }
        _tambah_baris(LOG_PATH, LOG_HEADERS, utama)

        detail = dict(data)
        detail["no"] = no
        _tambah_baris(DETAIL_PATH, DETAIL_HEADERS, detail)

    @staticmethod
    def _jumlah_baris_utama():
        if not os.path.exists(LOG_PATH):
            return 0
        with open(LOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
            return max(sum(1 for _ in csv.reader(f)) - 1, 0)

    @staticmethod
    def _baca_csv(path):
        if not os.path.exists(path):
            return []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            return [row for row in csv.DictReader(f) if row]

    def _baca_semua_baris_log(self):
        """Baris file utama digabung dengan file detail (lewat kolom 'no')."""
        detail = {r.get("no"): r for r in self._baca_csv(DETAIL_PATH)}
        gabungan = []
        for r in self._baca_csv(LOG_PATH):
            if (r.get("no") or "").strip().lower() == "no":
                continue  # header duplikat
            baris = dict(detail.get(r.get("no"), {}))
            baris.update(r)
            gabungan.append(baris)
        return gabungan

    def _baca_baris_dievaluasi(self):
        """Seluruh scan di log (terbaca maupun tidak terbaca)."""
        return self._baca_semua_baris_log()

    def perbarui_label_csv(self):
        """
        Hitung ulang kolom hasil_* di evaluation_log.csv dari kolom mentah
        (barcode_asli, ada_barcode, pyzbar, vision, final). Berguna setelah
        ada_barcode / barcode_asli diedit manual di Excel.
        """
        rows = self._baca_csv(LOG_PATH)
        if not rows:
            return False

        for r in rows:
            ada = self._ada_dari_baris(r)
            r["ada_barcode"] = _label_ada(ada)  # isi sel yang masih kosong agar CSV jelas
            for method, kunci_pred in PRED_KEYS.items():
                r[HASIL_KEYS[method]] = self.hitung_label(r.get(kunci_pred), r.get("barcode_asli"), ada)

        try:
            _tulis_csv(LOG_PATH, LOG_HEADERS, [[r.get(h, "") or "" for h in LOG_HEADERS] for r in rows])
        except PermissionError:
            print("[LiveEvaluator] evaluation_log.csv sedang dibuka (mis. di Excel), "
                  "kolom hasil_* tidak diperbarui. Perhitungan laporan tetap memakai data terbaru.")
            return False
        return True

    # ------------------------------------------------------------------
    # Helper statis
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value):
        """Konversi nilai CSV menjadi float; None jika kosong/tidak valid."""
        if value in (None, "", "None", "null"):
            return None
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _normalisasi_barcode(value):
        if value is None:
            return ""
        value = str(value).strip()
        if value.lower() in ("", "none", "null", "nan"):
            return ""
        # Jangan konversi ke int/float karena barcode dapat memiliki nol di depan.
        return value

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in ("true", "1", "yes", "ya", "ada", "positive", "positif"):
            return True
        if text in ("false", "0", "no", "tidak", "tidak ada", "negative", "negatif"):
            return False
        return None

    @classmethod
    def _ada_dari_baris(cls, row):
        """
        Baca kolom ada_barcode (Ya/Tidak) -> True/False.
        Bila kosong, diturunkan dari barcode_asli (terisi = ada, kosong = tidak ada),
        sehingga SEMUA scan ikut dievaluasi.
        """
        ada = cls._parse_bool(row.get("ada_barcode"))
        if ada is None:
            ada = bool(cls._normalisasi_barcode(row.get("barcode_asli")))
        return ada

    @classmethod
    def hitung_label(cls, prediksi, barcode_asli, ada_barcode):
        """
        Label satu sampel: 'TP' / 'TN' / 'FP' / 'FN', atau '' bila belum bisa dinilai.
          ada_barcode True : prediksi kosong -> FN; sama dengan asli -> TP; beda -> FP
          ada_barcode False: prediksi kosong -> TN; terisi (menebak) -> FP
          ada_barcode None : belum dilabeli -> ''
        """
        if ada_barcode is None:
            return ""

        pred = cls._normalisasi_barcode(prediksi)
        asli = cls._normalisasi_barcode(barcode_asli)

        if ada_barcode:
            if not pred:
                return "FN"
            if not asli:
                return ""  # ada barcode, tapi angka aslinya belum diisi
            return "TP" if pred == asli else "FP"

        return "FP" if pred else "TN"

    @staticmethod
    def _levenshtein(a, b):
        """Jarak edit (edit distance) antar 2 string, algoritma DP standar."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr_row = [i] + [0] * len(b)
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr_row[j] = min(
                    prev_row[j] + 1,
                    curr_row[j - 1] + 1,
                    prev_row[j - 1] + cost,
                )
            prev_row = curr_row
        return prev_row[-1]

    @staticmethod
    def _format_latency(detik):
        """Tampilkan dalam ms kalau sangat kecil, agar tidak terkesan "0.0 detik"."""
        if detik is None:
            return "-"
        if detik < 0.01:
            return f"{detik * 1000:.3f} ms"
        return f"{detik:.2f} detik"

    @staticmethod
    def _cetak_tabel(headers, rows):
        lebar = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                lebar[i] = max(lebar[i], len(str(cell)))

        def format_row(cells):
            return "  " + " | ".join(str(c).ljust(lebar[i]) for i, c in enumerate(cells))

        print(format_row(headers))
        print("  " + "-+-".join("-" * w for w in lebar))
        for row in rows:
            print(format_row(row))

    # ------------------------------------------------------------------
    # Metrik instan (per scan)
    # ------------------------------------------------------------------

    def compute_akurasi_filter_kategori(self, product, similar_products):
        """
        Persentase produk pembanding yang cocok pada jenis_spesifik DAN satuan.
        Jika metadata target tidak lengkap, hasil tidak dihitung (None), bukan
        menganggap semua pembanding benar.
        """
        if not similar_products or not isinstance(product, dict):
            return None

        jenis = product.get("jenis_spesifik")
        satuan = product.get("satuan")
        if not jenis or not satuan:
            return None

        valid_count = sum(
            1 for item in similar_products
            if isinstance(item, dict)
            and item.get("jenis_spesifik") == jenis
            and item.get("satuan") == satuan
        )
        return round(valid_count / len(similar_products) * 100, 1)

    def compute_context_recall_estimate(self, total_valid_di_db, jumlah_ditampilkan):
        """Estimasi context recall dari perbandingan jumlah pembanding di DB vs yang ditampilkan."""
        if not total_valid_di_db:
            return None
        recall = min(jumlah_ditampilkan, total_valid_di_db) / total_valid_di_db
        return round(recall * 100, 1)

    def compute_agreement(self, barcode_pyzbar, barcode_vision):
        if not barcode_pyzbar or not barcode_vision:
            return "Tidak dapat dibandingkan"
        return "Sama" if barcode_pyzbar == barcode_vision else "Beda"

    def compute_kebenaran(self, barcode_prediksi, barcode_asli):
        """Bandingkan prediksi dengan ground truth. Return True/False/None (None = tanpa ground truth)."""
        if not barcode_asli:
            return None
        if not barcode_prediksi:
            return False
        return str(barcode_prediksi).strip() == str(barcode_asli).strip()

    def compute_cer(self, barcode_prediksi, barcode_asli):
        """
        Character Error Rate per sampel: edit distance / panjang ground truth.
        CER=0 berarti cocok persis; prediksi kosong untuk barcode yang ada = 1.
        """
        if barcode_asli in (None, "", "None"):
            return None

        barcode_asli = str(barcode_asli).strip()
        if not barcode_asli:
            return None

        barcode_prediksi = "" if barcode_prediksi in (None, "None") else str(barcode_prediksi).strip()
        dist = self._levenshtein(barcode_prediksi, barcode_asli)
        return round(dist / len(barcode_asli), 6)

    def compute_estimasi_biaya(self, token_llm_explainer):
        if token_llm_explainer in (None, "", "None"):
            return None
        return round((float(token_llm_explainer) / 1000) * self.harga_per_1000_token, 2)

    # ------------------------------------------------------------------
    # Cetak ke terminal
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_benar(benar, cer):
        teks = _ya_tidak(benar)
        if cer is not None:
            teks += f"  (CER: {cer})"
        return teks

    def print_summary(self, data):
        lebar_label = 30

        def baris(label, value):
            print(f"   {label:<{lebar_label}}: {value}")

        cd = data.get("kondisi_debug") or {}

        print()
        print(f"📷 SCAN  |  Mode: {data['mode']}  |  {data['timestamp']}  |  no {data.get('no', '-')}")

        print("🔹 1. Deteksi & Perbandingan Barcode")
        baris("Barcode pyzbar", data["barcode_pyzbar"] or "-")
        baris("Barcode Vision AI", data["barcode_vision_tebakan"] or "-")
        baris("Barcode dipakai (lookup)", data["barcode_dipakai"] or "-")
        baris("Agreement pyzbar vs Vision", data["agreement_pyzbar_vision"])

        if data["barcode_asli"]:
            baris("Barcode asli (ground truth)", data["barcode_asli"])
            baris("pyzbar benar?", self._fmt_benar(data["pyzbar_benar"], data.get("cer_pyzbar")))
            baris("Vision AI benar?", self._fmt_benar(data["vision_benar"], data.get("cer_vision")))
        else:
            baris("Ground truth", "- (tidak diisi, normal lewat app.py)")

        if data.get("ada_barcode"):
            baris("Ada barcode di gambar?", data["ada_barcode"])
            baris(
                "Label pyzbar / vision / final",
                f"{data.get('hasil_pyzbar') or '-'} / {data.get('hasil_vision') or '-'} / "
                f"{data.get('hasil_final') or '-'}",
            )
        else:
            baris("Label TP/TN/FP/FN", "- (ada_barcode belum diisi; isi manual di CSV)")

        print("🔹 2. Validasi Struktural")
        baris("Checksum pyzbar valid?", _ya_tidak(data.get("pyzbar_checksum_valid")))
        baris("Checksum Vision AI valid?", _ya_tidak(data.get("vision_checksum_valid")))

        print("🔹 3. Efisiensi & Biaya")
        crop = data.get("gambar_di_crop")
        if crop is not None:
            baris("Gambar Vision AI di-crop?", "Ya (hemat token)" if crop else "Tidak (fallback penuh)")
        biaya = data.get("estimasi_biaya_rupiah")
        token_llm = data["token_llm_explainer"] or "tidak tersedia"
        baris("Token LLM Explainer", f"{token_llm}{'' if biaya is None else f'  (≈ Rp {biaya})'}")
        baris("Latency pyzbar", self._format_latency(data["latency_pyzbar_detik"]))
        baris("Latency Vision AI", self._format_latency(data["latency_vision_detik"]))
        baris("Latency LLM Explainer", self._format_latency(data["latency_llm_detik"]))
        baris("Latency Total", self._format_latency(data["latency_total_detik"]))

        print("🔹 4. Kualitas Retrieval Database")
        baris("Produk ditemukan?", "Ya" if data["produk_ditemukan"] else "Tidak")
        if data["produk_ditemukan"]:
            baris(
                "Pembanding tampil / valid di DB",
                f"{data['jumlah_pembanding_ditampilkan']} / {data['jumlah_pembanding_valid_di_db']}",
            )
            baris("Cakupan pembanding (estimasi)", fmt_pct(data["context_recall_estimate_persen"]))
            baris("Akurasi Filter Kategori", fmt_pct(data["akurasi_filter_kategori_persen"]))

        print("🔹 5. Kondisi Pengujian (auto-deteksi)")
        if data.get("kondisi_pencahayaan"):
            ket = f"(brightness V-HSV: {cd.get('brightness')}/255)" if cd else ""
            baris("Pencahayaan", f"{data['kondisi_pencahayaan']}  {ket}")
        if data.get("kondisi_fisik_barcode"):
            ket = (
                f"(skor buruk: {cd.get('skor_buruk')}/4"
                f" -- kontras={cd.get('kontras')}, ketajaman={cd.get('ketajaman_laplacian')})"
                if cd else ""
            )
            baris("Fisik barcode", f"{data['kondisi_fisik_barcode']}  {ket}")
        if data.get("image_detail_level"):
            baris("Image detail level", f"{data['image_detail_level']}  (parameter tetap)")

        self._print_running_tally()
        print()

    def _print_running_tally(self):
        """Ringkasan kumulatif dari seluruh histori scan di evaluation_log.csv."""
        rows = self._baca_semua_baris_log()
        if not rows:
            return

        total = len(rows)
        ditemukan = sum(1 for r in rows if str(r.get("produk_ditemukan")).strip() == "True")
        coverage = round(ditemukan / total * 100, 1)

        print(f"  Total scan sejauh ini: {total}")
        print(f"  Rasio produk ditemukan: {ditemukan}/{total} ({coverage}%)")

        for label, kunci in (("Pyzbar", "pyzbar"), ("Vision AI", "vision"), ("Final", "final")):
            m = self._hitung_matriks(rows, kunci)
            print(
                f"  {label} (dari {m['n_sample']} scan): "
                f"TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']} "
                f"-> Accuracy {fmt_pct(m['accuracy_persen'])}"
            )

    # ------------------------------------------------------------------
    # Ringkasan efisiensi
    # ------------------------------------------------------------------

    def _hitung_ringkasan_efisiensi(self):
        """Rata-rata latency, rata-rata token, total biaya, dan jumlah pengujian dari seluruh log."""
        rows = self._baca_semua_baris_log()
        if not rows:
            return None

        def kolom_float(nama):
            nilai = (self._safe_float(r.get(nama)) for r in rows)
            return [v for v in nilai if v is not None]

        latency_list = kolom_float("latency_detik")
        token_list = kolom_float("token")
        biaya_list = kolom_float("estimasi_biaya_rupiah")

        return {
            "rata_rata_latency_detik": round(sum(latency_list) / len(latency_list), 2) if latency_list else None,
            "rata_rata_token": round(sum(token_list) / len(token_list), 1) if token_list else None,
            "estimasi_biaya_rupiah": round(sum(biaya_list), 2) if biaya_list else None,
            "jumlah_pengujian": len(rows),
        }

    def print_ringkasan_efisiensi(self):
        """
        Cetak tabel ringkasan efisiensi dari seluruh histori scan.
        Sengaja tidak dipanggil otomatis per-scan, hanya lewat
        run_full_evaluation.py saat mau melihat laporan.
        """
        r = self._hitung_ringkasan_efisiensi()
        if not r:
            return

        baris_tabel = [
            ("Rata-rata Latency",
             f"{r['rata_rata_latency_detik']} detik" if r["rata_rata_latency_detik"] is not None else "-"),
            ("Rata-rata Total Token",
             f"{r['rata_rata_token']} token" if r["rata_rata_token"] is not None else "-"),
            ("Estimasi Biaya LLM Explainer",
             f"Rp {r['estimasi_biaya_rupiah']}" if r["estimasi_biaya_rupiah"] is not None else "-"),
            ("Jumlah Pengujian", str(r["jumlah_pengujian"])),
        ]

        print()
        print("  Ringkasan Efisiensi (kumulatif, seluruh histori scan):")
        self._cetak_tabel(["Parameter", "Nilai"], baris_tabel)

    # ------------------------------------------------------------------
    # Evaluasi per scan
    # ------------------------------------------------------------------

    def evaluate_and_report(
        self,
        mode,
        barcode_pyzbar,
        barcode_vision_tebakan,
        barcode_dipakai,
        product,
        similar_products,
        total_valid_di_db,
        token_llm_explainer,
        latency_vision_detik,
        latency_llm_detik,
        context_text,
        ai_answer_text,
        token_vision_ai=None,
        barcode_asli=None,
        vision_confidence=None,             # tidak lagi dicatat (selalu kosong); dipertahankan agar pemanggil lama tetap jalan
        latency_pyzbar_detik=0.0,
        kondisi_pencahayaan=None,           # "terang" / "redup" -- hasil deteksi otomatis
        kondisi_fisik_barcode=None,         # "mulus" / "rusak_pudar" -- hasil deteksi otomatis
        image_detail_level=None,            # "low" / "high" -- parameter tetap Vision AI
        kondisi_debug=None,                 # dict mentah dari condition_detector
        pyzbar_checksum_valid=None,         # hasil validate_checksum() untuk pyzbar
        vision_checksum_valid=None,         # hasil validate_checksum() untuk Vision AI
        gambar_di_crop=None,                # True/False -- gambar hasil crop atau tidak
        ground_truth_barcode_present=None,  # True/False; kosong = otomatis dari barcode_asli
    ):
        # ada_barcode: pakai nilai eksplisit dari caller bila ada; kalau tidak,
        # otomatis dari barcode_asli (terisi = ada, kosong = tanpa barcode),
        # sehingga setiap scan ikut dievaluasi.
        asli_norm = self._normalisasi_barcode(barcode_asli)
        ada = ground_truth_barcode_present
        if ada is None:
            ada = bool(asli_norm)

        produk_ditemukan = product is not None
        jumlah_ditampilkan = len(similar_products) if similar_products else 0

        akurasi_filter = context_recall = None
        if produk_ditemukan:
            akurasi_filter = self.compute_akurasi_filter_kategori(product, similar_products)
            context_recall = self.compute_context_recall_estimate(total_valid_di_db, jumlah_ditampilkan)

        latency_pyzbar_detik = latency_pyzbar_detik or 0.0
        estimasi_biaya = self.compute_estimasi_biaya(token_llm_explainer)

        token_vision_ai = _int_atau_none(token_vision_ai)
        token_llm_explainer = _int_atau_none(token_llm_explainer)

        # Total token hanya dihitung bila penggunaan token kedua tahap diketahui.
        # Jika suatu tahap memang tidak dipanggil, caller sebaiknya mengirim 0;
        # None berarti datanya tidak tersedia sehingga total tidak boleh ditebak.
        token_total = None
        if token_vision_ai is not None and token_llm_explainer is not None:
            token_total = token_vision_ai + token_llm_explainer

        latency_total = round(
            float(latency_pyzbar_detik or 0)
            + float(latency_vision_detik or 0)
            + float(latency_llm_detik or 0),
            6,
        )

        data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "barcode_pyzbar": barcode_pyzbar,
            "barcode_vision_tebakan": barcode_vision_tebakan,
            "barcode_dipakai": barcode_dipakai,
            "barcode_asli": barcode_asli,
            "ada_barcode": _label_ada(ada),
            "hasil_pyzbar": self.hitung_label(barcode_pyzbar, asli_norm, ada),
            "hasil_vision": self.hitung_label(barcode_vision_tebakan, asli_norm, ada),
            "hasil_final": self.hitung_label(barcode_dipakai, asli_norm, ada),
            "agreement_pyzbar_vision": self.compute_agreement(barcode_pyzbar, barcode_vision_tebakan),
            "pyzbar_benar": self.compute_kebenaran(barcode_pyzbar, barcode_asli),
            "vision_benar": self.compute_kebenaran(barcode_vision_tebakan, barcode_asli),
            "cer_pyzbar": self.compute_cer(barcode_pyzbar, barcode_asli),
            "cer_vision": self.compute_cer(barcode_vision_tebakan, barcode_asli),
            "pyzbar_checksum_valid": pyzbar_checksum_valid,
            "vision_checksum_valid": vision_checksum_valid,
            "gambar_di_crop": gambar_di_crop,
            "kondisi_pencahayaan": kondisi_pencahayaan,
            "kondisi_fisik_barcode": kondisi_fisik_barcode,
            "image_detail_level": image_detail_level,
            "kondisi_debug": kondisi_debug,  # tidak di-log ke CSV (hanya untuk tampilan terminal)
            "produk_ditemukan": produk_ditemukan,
            "jumlah_pembanding_ditampilkan": jumlah_ditampilkan,
            "jumlah_pembanding_valid_di_db": total_valid_di_db,
            "context_recall_estimate_persen": context_recall,
            "akurasi_filter_kategori_persen": akurasi_filter,
            "token_vision_ai": token_vision_ai,
            "token_llm_explainer": token_llm_explainer,
            "token_total": token_total,
            "estimasi_biaya_rupiah": estimasi_biaya,
            "latency_pyzbar_detik": latency_pyzbar_detik,
            "latency_vision_detik": latency_vision_detik,
            "latency_llm_detik": latency_llm_detik,
            "latency_total_detik": latency_total,
            "context_text": context_text,
            "ai_answer_text": ai_answer_text,
        }

        self.log_to_csv(data)
        self.print_summary(data)
        return data

    # ------------------------------------------------------------------
    # Metrik batch/agregat (dari evaluation_log.csv, dipanggil setelah
    # data testing berlabel terkumpul -- bukan per-scan)
    # ------------------------------------------------------------------

    @classmethod
    def _hitung_matriks(cls, rows, kunci_pred):
        """
        TP/TN/FP/FN + Accuracy/Precision/Recall/F1 untuk satu metode.
        Label dihitung ulang dari kolom mentah (barcode_asli, ada_barcode, prediksi),
        sehingga hasil tetap benar walau CSV diedit manual.
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        tp = tn = fp = fn = 0

        for r in rows:
            label = cls.hitung_label(r.get(kunci_pred), r.get("barcode_asli"), cls._ada_dari_baris(r))
            if label == "TP":
                tp += 1
            elif label == "TN":
                tn += 1
            elif label == "FP":
                fp += 1
            elif label == "FN":
                fn += 1

        n = tp + tn + fp + fn
        accuracy = (tp + tn) / n if n else None
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "n_sample": n,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy_persen": round(accuracy * 100, 1) if accuracy is not None else None,
            "precision_persen": round(precision * 100, 1),
            "recall_persen": round(recall * 100, 1),
            "f1_score_persen": round(f1 * 100, 1),
            "metric_type": "label_tp_tn_fp_fn",
        }

    def _hitung_cer_agregat(self, rows, prediction_key):
        """CER korpus = total edit distance / total karakter ground truth."""
        total_distance = 0
        total_reference_chars = 0
        per_sample = []

        for row in rows:
            actual = self._normalisasi_barcode(row.get("barcode_asli"))
            if not actual:
                continue
            pred = self._normalisasi_barcode(row.get(prediction_key))
            distance = self._levenshtein(pred, actual)
            total_distance += distance
            total_reference_chars += len(actual)
            per_sample.append(distance / len(actual))

        if total_reference_chars == 0:
            return {"cer": None, "cer_mean_per_sample": None, "n_sample": 0}

        return {
            "cer": round(total_distance / total_reference_chars, 6),
            "cer_mean_per_sample": round(sum(per_sample) / len(per_sample), 6),
            "n_sample": len(per_sample),
        }

    def compute_batch_accuracy(self):
        """
        Evaluasi batch: satu tabel confusion matrix (TP/TN/FP/FN) per metode,
        dihitung dari SELURUH scan di log (terbaca maupun tidak terbaca).
        """
        semua = self._baca_semua_baris_log()
        if not semua:
            print("Log masih kosong.")
            return None

        self.perbarui_label_csv()

        rows = semua  # seluruh scan dievaluasi, terbaca maupun tidak terbaca

        n_ada = sum(1 for r in rows if self._ada_dari_baris(r))
        n_tidak = len(rows) - n_ada

        print("\n" + "=" * 90)
        print(f"[BATCH] Confusion matrix pembacaan barcode (n={len(rows)} scan: "
              f"{n_ada} ada barcode, {n_tidak} tanpa barcode)")
        print("TP=terbaca benar | TN=tanpa barcode & tidak terbaca | "
              "FP=salah angka / menebak | FN=ada barcode tapi tidak terbaca")
        print("Accuracy = (TP + TN) / (TP + TN + FP + FN)")
        print("=" * 90)

        hasil = {}
        table_rows = []
        for method, key in PRED_KEYS.items():
            metrics = self._hitung_matriks(rows, key)
            cer = self._hitung_cer_agregat(rows, key)
            metrics["CER"] = cer["cer"]
            metrics["CER_mean_per_sample"] = cer["cer_mean_per_sample"]
            hasil[method] = metrics
            table_rows.append((
                method, metrics["n_sample"],
                metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"],
                fmt_pct(metrics["accuracy_persen"]), fmt_pct(metrics["precision_persen"]),
                fmt_pct(metrics["recall_persen"]), fmt_pct(metrics["f1_score_persen"]),
                "-" if metrics["CER"] is None else f"{metrics['CER']:.6f}",
            ))

        self._cetak_tabel(
            ["Metode", "n", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1", "CER korpus"],
            table_rows,
        )

        if n_tidak:
            print(
                f"\nCatatan: {n_tidak} scan berstatus 'tanpa barcode' (barcode_asli kosong). "
                "Bila barcode-nya sebenarnya ada di kemasan tapi tidak terbaca, isi barcode_asli "
                "(atau ada_barcode = Ya) di CSV agar terhitung FN, bukan TN."
            )

        # final kosong berarti pyzbar dan vision juga kosong, jadi cukup cek 'final'
        tak_bisa = sum(
            1 for r in rows
            if self.hitung_label(r.get("final"), r.get("barcode_asli"), self._ada_dari_baris(r)) == ""
        )
        if tak_bisa:
            print(f"Catatan: {tak_bisa} scan tidak bisa dinilai (ada_barcode = Ya tapi barcode_asli kosong).")

        # Alias untuk kode lama yang membaca hasil["barcode_presence_detection"]
        hasil["barcode_presence_detection"] = {m: hasil[m] for m in PRED_KEYS}
        return hasil

    def compute_robustness_breakdown(self):
        """
        Pecah TP/TN/FP/FN & accuracy pyzbar dan Vision AI berdasarkan kondisi
        pengujian otomatis (pencahayaan, fisik barcode, detail level).
        Kolom kondisi dibaca dari evaluation_detail.csv.
        """
        rows = self._baca_baris_dievaluasi()
        if not rows:
            print("Log masih kosong.")
            return None

        dimensi = {kolom: defaultdict(list) for kolom in KOLOM_KONDISI}

        for r in rows:
            for kolom in KOLOM_KONDISI:
                dimensi[kolom][r.get(kolom) or "tidak_diketahui"].append(r)

        hasil_akhir = {}
        print("\n[BATCH] Breakdown Robustness per Kondisi Pengujian")

        for kolom, kelompok in dimensi.items():
            print(f"\n  >> Berdasarkan {kolom}:")
            hasil_akhir[kolom] = {}
            baris_tabel = []

            for nilai_kondisi, bucket in kelompok.items():
                m_pyzbar = self._hitung_matriks(bucket, "pyzbar")
                m_vision = self._hitung_matriks(bucket, "vision")
                hasil_akhir[kolom][nilai_kondisi] = {"pyzbar": m_pyzbar, "vision_ai": m_vision}

                for nama, m in (("pyzbar", m_pyzbar), ("vision_ai", m_vision)):
                    baris_tabel.append((
                        f"{nilai_kondisi} (n={m['n_sample']})" if nama == "pyzbar" else "",
                        nama, m["TP"], m["TN"], m["FP"], m["FN"],
                        fmt_pct(m["accuracy_persen"]), fmt_pct(m["f1_score_persen"]),
                    ))

            self._cetak_tabel(
                ["Kondisi", "Metode", "TP", "TN", "FP", "FN", "Accuracy", "F1-score"],
                baris_tabel,
            )

        return hasil_akhir

    def compute_distribusi_kondisi(self):
        """
        Hitung distribusi jumlah scan per kondisi terdeteksi (pencahayaan,
        fisik barcode, detail level), tanpa butuh label sama sekali.
        """
        if not os.path.exists(LOG_PATH):
            print("Log belum ada.")
            return None

        rows = self._baca_semua_baris_log()
        if not rows:
            print("Log masih kosong.")
            return None

        hasil = {}
        print(f"\n[BATCH] Distribusi Kondisi Terdeteksi (n={len(rows)} scan, TANPA ground truth)")

        for kolom in KOLOM_KONDISI:
            counter = defaultdict(int)
            for r in rows:
                counter[r.get(kolom) or "tidak_diketahui"] += 1
            hasil[kolom] = dict(counter)

            print(f"\n  >> {kolom}:")
            for nilai, jumlah in counter.items():
                print(f"    - {nilai}: {jumlah}/{len(rows)} ({round(jumlah / len(rows) * 100, 1)}%)")
        print()

        return hasil