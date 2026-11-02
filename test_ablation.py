"""
test_ablation.py

Menjalankan Ablation Study A-D menggunakan
11 gambar yang tersimpan di:

marketrevisi/evaluation/dataset_images/

Konfigurasi:
A-Pyzbar
B-LVLM
C-Hybrid
D-Sistem Lengkap

Ground truth TIDAK dikirim ke Vision AI.
Ground truth hanya digunakan oleh evaluator
untuk menghitung Accuracy, Precision, Recall,
F1-Score, CER, dan Database Matching.
"""

import os
import re
import sys

from service.product_service import ProductService
from evaluation.ablation_evaluator import AblationEvaluator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lokasi: marketrevisi/test_ablation.py & marketrevisi/evaluation/dataset_images/
SCAN_DIR = os.path.join(BASE_DIR, "evaluation", "dataset_images")

CONFIGURATIONS = [
    "A-Pyzbar",
    "B-LVLM",
    "C-Hybrid",
    "D-Sistem Lengkap"
]


def clean_barcode(barcode):
    """Membersihkan barcode agar hanya berisi angka."""
    if barcode is None:
        return None

    barcode = str(barcode).strip()
    barcode = re.sub(r"\D", "", barcode)

    if barcode == "":
        return None

    return barcode


def extract_barcode_from_filename(filename):
    """
    Ambil deretan angka terakhir sebelum ekstensi sebagai ground truth.

    Contoh: 20260815_150518_testing_8886008101091.jpg -> 8886008101091

    Prefix timestamp (YYYYMMDD_HHMMSS_) dibuang dulu agar tanggal tidak ikut
    terbaca sebagai barcode. Gambar tanpa barcode (mis. ..._unknown.jpg)
    mengembalikan None -> dianggap sampel negatif (sumber TN).
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"^\d{8}_\d{6}_", "", name)
    matches = re.findall(r"\d{8,14}", name)

    if not matches:
        return None

    return clean_barcode(matches[-1])


def build_test_data():
    if not os.path.isdir(SCAN_DIR):
        raise FileNotFoundError(
            "\nFolder dataset_images tidak ditemukan:\n"
            f"{SCAN_DIR}\n\n"
            "Pastikan folder berada di:\n"
            f"{BASE_DIR}\\evaluation\\dataset_images"
        )

    files = []
    for filename in os.listdir(SCAN_DIR):
        lower = filename.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
            files.append(filename)

    files.sort()

    test_data = []
    for filename in files:
        barcode_asli = extract_barcode_from_filename(filename)

        if barcode_asli is None:
            # Gambar tanpa barcode -> sampel negatif: ground truth dikosongkan
            # agar evaluator menghitungnya sebagai TN/FP.
            print(f"[INFO] Sampel negatif (tanpa barcode di nama file): {filename}")
            barcode_asli = ""

        image_path = os.path.join(SCAN_DIR, filename)

        test_data.append({
            "image": image_path,
            "filename": filename,
            "barcode_asli": barcode_asli
        })

    return test_data


def validasi_test_data():
    print("\n" + "=" * 70)
    print("VALIDASI DATA ABLATION")
    print("=" * 70)
    print(f"Folder gambar       : {SCAN_DIR}")

    test_data = build_test_data()

    print(f"Jumlah gambar       : {len(test_data)}")
    print(f"Jumlah konfigurasi  : {len(CONFIGURATIONS)}")
    print(f"Total eksperimen    : {len(test_data) * len(CONFIGURATIONS)}")
    print("=" * 70)

    if len(test_data) == 0:
        raise RuntimeError(
            "\nTidak ada gambar yang dapat digunakan untuk Ablation Study."
        )

    print("\nDATA TESTING:")
    for i, item in enumerate(test_data, start=1):
        print(f"{i:02d}. {item['filename']}")
        print(f"    Ground Truth : {item['barcode_asli'] or '(tanpa barcode / negatif)'}")

    print("\n" + "=" * 70)

    return test_data


def jalankan_satu_eksperimen(service, evaluator, konfigurasi, item, nomor, total):
    image_path = item["image"]
    barcode_asli = item["barcode_asli"]

    print("\n" + "-" * 70)
    print(f"[{nomor}/{total}] {konfigurasi}")
    print("[INFO] Gambar:")
    print(f"       {item['filename']}")
    print("[INFO] Ground Truth:")
    print(f"       {barcode_asli or '(tanpa barcode / negatif)'}")
    print("-" * 70)

    try:
        # barcode_asli hanya diberikan ke evaluator/process_ablation(),
        # tidak dimasukkan ke prompt LVLM.
        hasil = service.process_ablation(
            image_path=image_path,
            barcode_asli=barcode_asli,
            konfigurasi=konfigurasi
        )

        evaluator.save_result(hasil)

        print("\n[HASIL]")
        print(f"Barcode Pyzbar : {hasil.get('barcode_pyzbar') or '-'}")
        print(f"Barcode Vision : {hasil.get('barcode_vision') or '-'}")
        print(f"Barcode Final  : {hasil.get('barcode_final') or '-'}")
        print(f"Ground Truth   : {barcode_asli or '-'}")
        print(f"Benar?         : {hasil.get('benar')}")
        print(f"CER            : {hasil.get('cer')}")
        print(f"Latency Total  : {hasil.get('latency_total', 0):.3f} detik")
        print(f"Token Vision   : {hasil.get('token_vision', 0)}")
        print(f"Token LLM      : {hasil.get('token_llm', 0)}")
        print(f"Token Total AI : {hasil.get('token_total_ai', 0)}")

        database_match = hasil.get("database_match")
        if database_match is not None:
            print(f"Database Match : {database_match}")

        return True

    except Exception as e:
        print("\n[ERROR]")
        print(f"Konfigurasi : {konfigurasi}")
        print(f"Gambar      : {item['filename']}")
        print(f"{type(e).__name__}: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("ABLATION STUDY TESTING")
    print("=" * 70)

    test_data = validasi_test_data()
    total_eksperimen = len(test_data) * len(CONFIGURATIONS)

    print("\n" + "=" * 70)
    print("KONFIRMASI EKSPERIMEN")
    print("=" * 70)
    print(f"Jumlah gambar       : {len(test_data)}")
    print(f"Jumlah konfigurasi  : {len(CONFIGURATIONS)}")
    print(f"Total eksperimen    : {total_eksperimen}")
    print("=" * 70)

    service = ProductService()
    evaluator = AblationEvaluator()

    print("\n[INFO] Menghapus hasil Ablation Study sebelumnya...")
    evaluator.reset()

    counter = 0
    berhasil = 0
    gagal = 0

    for konfigurasi in CONFIGURATIONS:
        print("\n" + "#" * 70)
        print(f"# KONFIGURASI: {konfigurasi}")
        print("#" * 70)

        for item in test_data:
            counter += 1

            sukses = jalankan_satu_eksperimen(
                service=service,
                evaluator=evaluator,
                konfigurasi=konfigurasi,
                item=item,
                nomor=counter,
                total=total_eksperimen
            )

            if sukses:
                berhasil += 1
            else:
                gagal += 1

    print("\n" + "=" * 70)
    print("EKSPERIMEN SELESAI")
    print("=" * 70)
    print(f"Total eksperimen : {total_eksperimen}")
    print(f"Berhasil         : {berhasil}")
    print(f"Gagal            : {gagal}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("HASIL RINGKASAN ABLATION STUDY")
    print("=" * 70)

    evaluator.print_summary()

    print("\n" + "=" * 70)
    print("ABLATION STUDY SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()