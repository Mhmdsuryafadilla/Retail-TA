"""
evaluation/condition_detector.py

Modul ini mendeteksi 3 kondisi pengujian secara otomatis dari gambar,
tanpa input manual dari tester:

1. Kondisi pencahayaan (terang/redup)
   -> dari rata-rata brightness (channel V di HSV) seluruh frame.

2. Kondisi fisik barcode (mulus/rusak_pudar)
   -> dari kontras & ketajaman (variance Laplacian) di area sekitar
      barcode (crop bbox hasil pyzbar kalau ada), dikombinasikan
      dengan status keberhasilan pyzbar & validasi checksum.

3. Image detail level (low/high)
   -> SEKARANG hasil DERIVASI dari kondisi #1 dan #2 (bukan konstanta
      tetap lagi). Aturan: "high" kalau ADA salah satu dari pencahayaan
      redup ATAU fisik barcode rusak_pudar; "low" hanya kalau kedua
      kondisi bagus (terang + mulus). Lihat tentukan_image_detail_level().
      Nilai ini diteruskan ke vision/vision_api.py sebagai parameter
      image_detail_level saat memanggil identify_product().

CATATAN METODOLOGI: deteksi pencahayaan & fisik barcode di sini bersifat
estimasi heuristik berbasis image processing (thresholding brightness/
kontras/ketajaman), bukan model klasifikasi terlatih. Threshold-nya
mudah dikalibrasi lewat konstanta di bawah.
"""

import cv2
import numpy as np

# Konstanta threshold -- sesuaikan kalau hasil deteksi tidak cocok
# dengan kondisi asli di lapangan (kalibrasi lewat angka yang tercetak
# di terminal saat scan).

# Brightness (rata-rata channel V, skala 0-255) di bawah nilai ini = "redup"
THRESHOLD_BRIGHTNESS_REDUP = 155

# Kontras (std dev grayscale) di bawah nilai ini = "pudar"/washed out
THRESHOLD_KONTRAS_PUDAR = 40

# Ketajaman (variance Laplacian) di bawah nilai ini = "blur"
THRESHOLD_KETAJAMAN_BLUR = 80

def tentukan_image_detail_level(kondisi_pencahayaan, kondisi_fisik_barcode):
    """
    Tentukan image_detail_level ("low"/"high") secara dinamis
    berdasarkan kondisi pencahayaan & fisik barcode.

    Aturan (Opsi A -- prioritas akurasi):
        - "high" kalau ADA SALAH SATU kondisi yang bermasalah
          (redup ATAU rusak_pudar), karena detail tinggi membantu
          Vision AI mengkompensasi gambar yang kurang ideal.
        - "low" HANYA kalau kedua kondisi bagus
          (terang DAN mulus), karena gambar sudah cukup jelas
          tanpa perlu detail tambahan (hemat token).

    Return: "low" atau "high"
    """
    ada_masalah = (
        kondisi_pencahayaan == "redup"
        or kondisi_fisik_barcode == "rusak_pudar"
    )
    return "high" if ada_masalah else "low"


def detect_kondisi_pencahayaan(frame_bgr):
    """Return: ("terang" / "redup", brightness_value)"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    kondisi = "terang" if brightness >= THRESHOLD_BRIGHTNESS_REDUP else "redup"
    return kondisi, round(brightness, 1)


def detect_kondisi_fisik_barcode(frame_bgr, bbox, found_pyzbar, checksum_valid):
    """
    Return: ("mulus" / "rusak_pudar", dict_metrics_untuk_debug)

    bbox           : (x, y, w, h) hasil deteksi pyzbar, atau None kalau gagal.
    found_pyzbar   : True/False, apakah pyzbar berhasil baca barcode.
    checksum_valid : True/False/None, hasil validasi checksum EAN/UPC.
    """
    if bbox is not None:
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        crop = frame_bgr[y:y + h, x:x + w]
        area_diperiksa = "area sekitar barcode (crop bbox hasil pyzbar)"
    else:
        crop = frame_bgr
        area_diperiksa = "seluruh gambar (fallback -- bbox tidak tersedia)"

    if crop is None or crop.size == 0:
        crop = frame_bgr
        area_diperiksa = "seluruh gambar (fallback -- crop kosong)"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    kontras = float(np.std(gray))
    ketajaman = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    pudar = kontras < THRESHOLD_KONTRAS_PUDAR
    blur = ketajaman < THRESHOLD_KETAJAMAN_BLUR
    pyzbar_gagal = not found_pyzbar
    checksum_invalid = checksum_valid is False

    # Makin banyak indikator negatif menyala, makin besar kemungkinan
    # kondisi fisik barcode memang rusak/pudar
    skor_buruk = sum([pudar, blur, pyzbar_gagal, checksum_invalid])

    # PENTING: pudar/blur adalah pengukuran VISUAL langsung terhadap
    # kondisi fisik barcode. pyzbar_gagal/checksum_invalid adalah sinyal
    # KEGAGALAN DECODE -- bisa terjadi walau barcode fisiknya mulus
    # (misal sudut kamera miring, jarak salah, tipe barcode tak
    # didukung, dsb). Karena itu "rusak_pudar" WAJIB didukung minimal
    # satu bukti visual langsung (pudar atau blur), tidak boleh
    # divonis hanya dari kegagalan decode saja.
    kualitas_visual_buruk = pudar or blur
    kondisi = "rusak_pudar" if (kualitas_visual_buruk and skor_buruk >= 2) else "mulus"

    metrics = {
        "area_diperiksa": area_diperiksa,
        "kontras": round(kontras, 1),
        "ketajaman_laplacian": round(ketajaman, 1),
        "pudar": pudar,
        "blur": blur,
        "pyzbar_gagal": pyzbar_gagal,
        "checksum_invalid": checksum_invalid,
        "skor_buruk": skor_buruk,
        "kualitas_visual_buruk": kualitas_visual_buruk,  # BARU, untuk debug
    }
    return kondisi, metrics


def detect_semua_kondisi(frame_bgr, bbox, found_pyzbar, checksum_valid):
    """
    Fungsi gabungan -- panggil sekali dari product_service.py.

    Return: (hasil, debug_metrics)
        hasil         : dict siap pakai untuk evaluate_and_report()
        debug_metrics : angka mentah di balik keputusan, untuk debug
    """
    kondisi_pencahayaan, brightness = detect_kondisi_pencahayaan(frame_bgr)
    kondisi_fisik_barcode, fisik_metrics = detect_kondisi_fisik_barcode(
        frame_bgr, bbox, found_pyzbar, checksum_valid
    )

    # Image detail level sekarang DINAMIS -- ditentukan dari kombinasi
    # kondisi pencahayaan & fisik barcode, bukan nilai tetap lagi.
    image_detail_level = tentukan_image_detail_level(
        kondisi_pencahayaan, kondisi_fisik_barcode
    )

    hasil = {
        "kondisi_pencahayaan": kondisi_pencahayaan,
        "kondisi_fisik_barcode": kondisi_fisik_barcode,
        "image_detail_level": image_detail_level,
    }

    debug_metrics = {
        "brightness": brightness,
        **fisik_metrics,
    }

    return hasil, debug_metrics


def print_debug_kondisi(hasil, debug_metrics):
    """Cetak hasil deteksi otomatis ke terminal, supaya tester bisa lihat langsung."""
    print("[AUTO-DETEKSI KONDISI] Hasil deteksi otomatis untuk scan ini:")
    print(f"  Kondisi pencahayaan      : {hasil['kondisi_pencahayaan']}  "
          f"(brightness={debug_metrics['brightness']}, threshold redup<{THRESHOLD_BRIGHTNESS_REDUP})")
    print(f"  Kondisi fisik barcode    : {hasil['kondisi_fisik_barcode']}")
    print(f"    - area diperiksa       : {debug_metrics['area_diperiksa']}")
    print(f"    - kontras              : {debug_metrics['kontras']} "
          f"(pudar? {debug_metrics['pudar']}, threshold<{THRESHOLD_KONTRAS_PUDAR})")
    print(f"    - ketajaman (laplacian): {debug_metrics['ketajaman_laplacian']} "
          f"(blur? {debug_metrics['blur']}, threshold<{THRESHOLD_KETAJAMAN_BLUR})")
    print(f"    - pyzbar gagal?        : {debug_metrics['pyzbar_gagal']}")
    print(f"    - checksum invalid?    : {debug_metrics['checksum_invalid']}")
    print(f"    - kualitas visual buruk?: {debug_metrics['kualitas_visual_buruk']} "
        f"(syarat WAJIB -- dari pudar/blur, bukan dari gagal decode)")
    print(f"    - skor buruk (0-4)     : {debug_metrics['skor_buruk']} "
        f"(>=2 DAN kualitas_visual_buruk -> rusak_pudar)")
    print(f"  Image detail level       : {hasil['image_detail_level']} "
        f"(dinamis: high jika ADA masalah pencahayaan/fisik, low jika keduanya bagus)")


def print_ringkas_kondisi(hasil, debug_metrics):
    """Cetak versi ringkas (3 baris) dari hasil deteksi otomatis."""

    baris_pencahayaan = (
        f"Kondisi pencahayaan (auto)     : {hasil['kondisi_pencahayaan']:<10}"
        f"(brightness V-HSV: {debug_metrics['brightness']}/255, "
        f"threshold redup <{THRESHOLD_BRIGHTNESS_REDUP})"
    )

    indikator_menyala = []
    if debug_metrics["pudar"]:
        indikator_menyala.append("pudar")
    if debug_metrics["blur"]:
        indikator_menyala.append("blur")
    if debug_metrics["pyzbar_gagal"]:
        indikator_menyala.append("pyzbar_gagal")
    if debug_metrics["checksum_invalid"]:
        indikator_menyala.append("checksum_invalid")

    keterangan_fisik = (
        ", ".join(indikator_menyala) if indikator_menyala else "tidak ada indikator buruk"
    )

    baris_fisik = (
        f"Kondisi fisik barcode (auto)   : {hasil['kondisi_fisik_barcode']:<10}"
        f"(skor buruk: {debug_metrics['skor_buruk']}/4, visual_buruk={debug_metrics['kualitas_visual_buruk']} "
        f"-- {keterangan_fisik})"
    )

    baris_detail = (
        f"Image detail level (Vision AI) : {hasil['image_detail_level']:<10}"
        f"(dinamis dari kondisi pencahayaan + fisik barcode)"
    )

    print("[RINGKASAN KONDISI]")
    print(f"  {baris_pencahayaan}")
    print(f"  {baris_fisik}")
    print(f"  {baris_detail}")