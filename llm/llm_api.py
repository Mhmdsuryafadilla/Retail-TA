"""
llm/llm_api.py

Panggil LLM (AI Explainer) untuk menjelaskan produk ke konsumen.
Pakai common/api_client.py untuk request HTTP (sama dengan vision_api.py)
supaya retry+backoff konsisten dan base URL provider cukup diatur di
1 tempat (config.py).
"""

import time

from config import VISION_MODEL  # 1 model dipakai untuk Vision & LLM Explainer
from common.api_client import post_chat_completion, APICallError


class LLMAPI:

    def __init__(self):
        # Info token & latency dari panggilan terakhir, dibaca dari
        # luar (product_service.py) untuk evaluasi
        self.last_usage = None
        self.last_latency_seconds = None

    def generate(self, prompt: str, max_retries: int = 3):

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Anda adalah AI Retail Assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }

        start = time.time()

        try:
            result = post_chat_completion(payload, max_retries=max_retries)
        except APICallError as e:
            print(f"[ERROR] LLM API gagal total: {e}")
            raise

        elapsed = time.time() - start

        self.last_usage = result.get("usage")
        self.last_latency_seconds = round(elapsed, 2)

        return result["choices"][0]["message"]["content"]