"""
vision/preprocessing.py

Satu langkah preprocessing yang HASILNYA DIPAKAI BERSAMA oleh pyzbar
maupun Vision AI (GPT-4.1-mini)
    Input Gambar/Kamera -> Preprocessing (Crop+Resize) -> [pyzbar, LVLM]


"""

import cv2


def preprocess_image(frame_bgr, max_width=800, center_crop_ratio=1.0):


    h, w = frame_bgr.shape[:2]

    if center_crop_ratio < 1.0:
        new_w = int(w * center_crop_ratio)
        new_h = int(h * center_crop_ratio)
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        frame_bgr = frame_bgr[y0:y0 + new_h, x0:x0 + new_w]
        h, w = frame_bgr.shape[:2]

    # ---- RESIZE (turunkan resolusi kalau lebih besar dari max_width) ----
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame_bgr


def crop_to_bbox(frame_bgr, bbox, padding=60, min_width=300):


    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = bbox

    # Tambah padding di semua sisi
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w_img, x + w + padding)
    y1 = min(h_img, y + h + padding)

  
    if (x1 - x0) < min_width:
        extra = (min_width - (x1 - x0)) // 2
        x0 = max(0, x0 - extra)
        x1 = min(w_img, x1 + extra)

    cropped = frame_bgr[y0:y1, x0:x1]

    if cropped.size == 0:
        return frame_bgr

    return cropped

def scale_bbox(bbox, from_width, to_width):
    """
    Skalakan bbox (x, y, w, h) dari frame dengan lebar `from_width`
    ke frame dengan lebar `to_width`.

    Dibutuhkan karena pyzbar sekarang dijalankan di frame resolusi TINGGI
    (supaya lebar bar barcode tidak hilang saat downscale), sedangkan
    Vision AI + deteksi kondisi jalan di frame resolusi lebih kecil.
    bbox hasil pyzbar harus di-scale dulu sebelum dipakai untuk crop
    di frame yang skalanya berbeda.
    """
    if from_width == 0:
        return bbox

    scale = to_width / from_width
    x, y, w, h = bbox

    return (
        int(round(x * scale)),
        int(round(y * scale)),
        int(round(w * scale)),
        int(round(h * scale)),
    )