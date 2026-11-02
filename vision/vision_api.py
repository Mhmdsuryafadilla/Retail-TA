"""
vision/vision_api.py
...
"""

import base64
import json

from config import VISION_MODEL
from common.api_client import post_chat_completion, APICallError

# Default detail level -- dipakai kalau caller tidak mengoper nilai lain.
# Ini SUMBER KEBENARAN untuk detail level Vision AI.
# evaluation/condition_detector.py meng-import nilai ini, bukan
# mendefinisikan ulang, supaya tidak ada dua tempat yang harus
# disinkronkan manual.
DEFAULT_IMAGE_DETAIL_LEVEL = "low"  # "low" atau "high"


class VisionAPI:

    def __init__(self):
        self.last_usage = None

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def identify_product(self, image_path, max_retries=3,
                          image_detail_level=None):

        self.last_usage = None

        detail_level = image_detail_level or DEFAULT_IMAGE_DETAIL_LEVEL
        if detail_level not in ("low", "high"):
            print(f"[WARNING] image_detail_level tidak valid: {detail_level!r}, "
                  f"fallback ke 'low'")
            detail_level = "low"

        image_base64 = self.encode_image(image_path)

        print(
            f"[DEBUG] Ukuran base64: {len(image_base64)} karakter "
            f"(~{len(image_base64) / 1024:.1f} KB)"
        )
        print(f"[DEBUG] Image detail level dipakai: {detail_level}")

        payload = {
            "model": VISION_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": """
Anda adalah AI Retail.

Identifikasi produk pada gambar.

Jika barcode terlihat, bacalah angka barcode dengan teliti.

Jawab HANYA dalam JSON dengan format:

{
    "nama_produk": "",
    "barcode": "",
    "merek": "",
    "confidence": 0
}

Jika barcode tidak terlihat atau tidak yakin,
isi barcode dengan string kosong.

Jangan mengarang barcode.
"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identifikasi produk ini."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": detail_level
                            }
                        }
                    ]
                }
            ]
        }

        try:
            result = post_chat_completion(payload, max_retries=max_retries)
        except APICallError as e:
            print(f"[ERROR] Vision API gagal total: {e}")
            raise

        self.last_usage = result.get("usage")
        if self.last_usage:
            print(f"[DEBUG] Vision AI usage: {self.last_usage}")
            print(f"[DEBUG] Token Vision AI: {self.last_usage.get('total_tokens', 0)}")

        content = result["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            print("[ERROR] Response Vision AI bukan JSON valid:")
            print(content)
            parsed = {
                "nama_produk": "",
                "barcode": "",
                "merek": "",
                "confidence": 0
            }

        return parsed