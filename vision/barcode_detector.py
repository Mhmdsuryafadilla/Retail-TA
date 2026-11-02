import cv2
from pyzbar import pyzbar


def detect_barcode(frame_bgr):
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
    except Exception as e:
        print(f"[DEBUG] EXCEPTION di detect_barcode: {e}")
        return False, None, None

    if not barcodes:
        return False, None, None

    barcode = barcodes[0]
    barcode_value = barcode.data.decode("utf-8")
    x, y, w, h = barcode.rect

    return True, barcode_value, (x, y, w, h)


def draw_barcode_box(frame_bgr, bbox, label=""):
    if bbox is None:
        return frame_bgr

    x, y, w, h = bbox
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if label:
        cv2.putText(
            frame_bgr, label, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    return frame_bgr