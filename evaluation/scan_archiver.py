"""
evaluation/scan_archiver.py

Modul ini punya 3 tugas:

1. save_scan_image()              -> simpan foto scan secara permanen
2. detect_brightness_condition()  -> hitung otomatis kondisi pencahayaan
   dari rata-rata piksel gambar (bukan tebakan manual tester)
3. render_kondisi_pengujian_ui()  -> form Streamlit untuk kondisi yang
   tidak bisa dihitung otomatis (kondisi fisik barcode) + ground truth
"""

import os
import time
import cv2
import streamlit as st

SCAN_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "scan_images")

# Ambang batas kecerahan (skala 0-255) -- sesuaikan dengan distribusi data uji
THRESHOLD_TERANG = 130   # rata-rata piksel >= ini -> "terang"
THRESHOLD_REDUP = 80     # rata-rata piksel < ini  -> "redup" (di antaranya -> "normal")


def _ensure_dir():
    os.makedirs(SCAN_IMAGES_DIR, exist_ok=True)


def detect_brightness_condition(image_source):
    """
    Hitung kondisi pencahayaan otomatis dari nilai piksel gambar
    (grayscale, dirata-ratakan, lalu dibandingkan ke threshold).

    image_source : path string atau frame OpenCV (numpy array BGR)

    Return: (kategori: "terang"/"normal"/"redup", nilai_rata_rata: float 0-255)
    """
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
    else:
        img = image_source

    if img is None:
        return "tidak_diketahui", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rata_rata = float(gray.mean())

    if rata_rata >= THRESHOLD_TERANG:
        kategori = "terang"
    elif rata_rata < THRESHOLD_REDUP:
        kategori = "redup"
    else:
        kategori = "normal"

    return kategori, round(rata_rata, 1)


def save_scan_image(image_source, barcode_hint=None, mode="unknown"):
    """Simpan satu foto scan secara permanen dengan nama file unik."""
    _ensure_dir()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_slug = mode.lower().replace(" ", "_")
    barcode_slug = barcode_hint if barcode_hint else "unknown"

    filename = f"{timestamp}_{mode_slug}_{barcode_slug}.jpg"
    filepath = os.path.join(SCAN_IMAGES_DIR, filename)

    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is not None:
            cv2.imwrite(filepath, img)
        else:
            with open(image_source, "rb") as src, open(filepath, "wb") as dst:
                dst.write(src.read())
    else:
        cv2.imwrite(filepath, image_source)

    return filepath


def render_kondisi_pengujian_ui(image_source, location=st.sidebar, key_prefix="", image_detail_level_config="low"):
    """
    Render form kondisi pengujian.
    - kondisi_pencahayaan dihitung otomatis dari image_source, ditampilkan
      untuk transparansi, dan boleh di-override manual oleh tester.
    - image_detail_level diambil dari konfigurasi sistem (parameter
      `detail` ke Vision AI), bukan dipilih manual.
    - kondisi_fisik_barcode tetap manual karena butuh penilaian visual
      yang sulit diukur otomatis dari piksel saja.

    image_source : path atau frame gambar yang sedang di-scan
    location      : st.sidebar (default) atau container lain
    key_prefix    : awalan unik untuk key widget (wajib beda tiap tab)
    image_detail_level_config : nilai `detail` yang dipakai di vision_api.py

    Return: dict
        {
            "barcode_asli": str atau None,
            "kondisi_pencahayaan": "terang" / "normal" / "redup",
            "kecerahan_rata_rata": float (0-255),
            "kondisi_fisik_barcode": "mulus" / "rusak_pudar" / None,
            "image_detail_level": str (otomatis dari config),
        }
    """
    kategori_otomatis, nilai_kecerahan = detect_brightness_condition(image_source)

    with location:
        st.markdown("**🧪 Kondisi Pengujian (opsional, untuk testing)**")

        mode_testing = st.checkbox(
            "Aktifkan mode testing (isi ground truth & kondisi)",
            key=f"{key_prefix}mode_testing",
            help="Aktifkan kalau sedang mengumpulkan data untuk evaluasi skripsi."
        )

        if not mode_testing:
            return {
                "barcode_asli": None,
                "kondisi_pencahayaan": None,
                "kecerahan_rata_rata": None,
                "kondisi_fisik_barcode": None,
                "image_detail_level": image_detail_level_config,
            }

        barcode_asli = st.text_input(
            "Barcode asli (ground truth)",
            key=f"{key_prefix}barcode_asli",
            help="Barcode yang BENAR-BENAR tertera di kemasan produk, dicek manual."
        )

        st.caption(
            f"💡 Kecerahan gambar terdeteksi otomatis: **{nilai_kecerahan}** "
            f"(skala 0-255) → dikategorikan **{kategori_otomatis}**"
        )

        opsi_pencahayaan = ["terang", "normal", "redup"]
        default_idx = opsi_pencahayaan.index(kategori_otomatis) if kategori_otomatis in opsi_pencahayaan else 1

        kondisi_pencahayaan = st.selectbox(
            "Kondisi pencahayaan (otomatis, boleh dikoreksi manual)",
            options=opsi_pencahayaan,
            index=default_idx,
            key=f"{key_prefix}kondisi_pencahayaan",
        )

        kondisi_fisik_barcode = st.selectbox(
            "Kondisi fisik barcode (penilaian visual manual)",
            options=["mulus", "rusak_pudar"],
            key=f"{key_prefix}kondisi_fisik_barcode",
            help="Barcode rusak/pudar/tertutup perlu dinilai langsung oleh mata, "
                 "belum bisa dihitung otomatis dari piksel saja."
        )

        return {
            "barcode_asli": barcode_asli.strip() if barcode_asli else None,
            "kondisi_pencahayaan": kondisi_pencahayaan,
            "kecerahan_rata_rata": nilai_kecerahan,
            "kondisi_fisik_barcode": kondisi_fisik_barcode,
            "image_detail_level": image_detail_level_config,
        }