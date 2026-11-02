"""
common/api_client.py

Klien HTTP terpusat untuk SEMUA panggilan ke API LLM/Vision (endpoint
chat completions yang OpenAI-compatible), dipakai bersama oleh
vision/vision_api.py DAN llm/llm_api.py.

KENAPA DISATUKAN DI SINI (bukan tiap file punya requests.post sendiri):

1. SATU TEMPAT untuk ganti provider. Kalau maiarouter.ai bermasalah
   lagi (down, akun diblokir, dll), kamu cukup ubah BASE_URL di
   config.py -- TIDAK PERLU edit vision_api.py dan llm_api.py satu-satu.

2. Retry + exponential backoff diterapkan KONSISTEN di kedua jalur.
   Sebelumnya cuma llm_api.py yang punya retry untuk 429 -- vision_api.py
   langsung crash begitu kena 429 (ini yang bikin app.py error dan
   proses scan berhenti total, padahal harusnya cuma retry lalu lanjut).

3. Membedakan error SEMENTARA (429 rate limit, 5xx server error,
   connection error -- masuk akal untuk di-retry) vs error PERMANEN
   (400 bad request, 401 unauthorized, 404 -- retry tidak akan
   mengubah apa-apa, jadi langsung dilempar tanpa buang waktu retry).
"""

import time
import requests

from config import OPENAI_API_KEY, BASE_URL

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}


class APICallError(Exception):
    """Dilempar kalau panggilan API gagal setelah semua percobaan retry habis."""
    pass


def post_chat_completion(payload, max_retries=3, base_delay_seconds=3, timeout=60):
    """
    POST ke {BASE_URL}/chat/completions, dengan retry + exponential
    backoff untuk error yang BERSIFAT SEMENTARA:
        - HTTP 429 (rate limit)
        - HTTP 5xx (server error di sisi provider)
        - Connection error (timeout, DNS gagal, koneksi putus, dll)

    Error PERMANEN (400, 401, 403, 404, dll) LANGSUNG dilempar tanpa
    retry, karena diulang pun hasilnya pasti sama persis.

    Return: dict hasil response.json() kalau sukses.
    Raise : APICallError kalau gagal setelah semua retry habis, dengan
            pesan yang menjelaskan apakah ini kemungkinan masalah di
            sisi provider atau di kode/konfigurasi kita.
    """
    url = f"{BASE_URL.rstrip('/')}/chat/completions"

    last_error_detail = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            # Error KONEKSI (timeout, DNS gagal, koneksi putus, dll) --
            # ini juga sering bersifat sementara, jadi ikut di-retry.
            last_error_detail = f"Connection error: {e}"
            print(f"[API RETRY {attempt}/{max_retries}] {last_error_detail}")
            if attempt < max_retries:
                delay = base_delay_seconds * attempt
                print(f"  Menunggu {delay} detik sebelum coba lagi...")
                time.sleep(delay)
                continue
            raise APICallError(
                f"Gagal konek ke {url} setelah {max_retries} percobaan. "
                f"Cek koneksi internet kamu. Detail: {last_error_detail}"
            )

        # ---- Error SEMENTARA (rate limit / server error) -> RETRY ----
        if response.status_code == 429 or response.status_code >= 500:
            last_error_detail = (
                f"HTTP {response.status_code} (percobaan {attempt}/{max_retries}). "
                f"Body: {response.text[:300]}"
            )
            print(f"[API RETRY {attempt}/{max_retries}] {last_error_detail}")

            if attempt < max_retries:
                delay = base_delay_seconds * attempt
                print(f"  Menunggu {delay} detik sebelum coba lagi...")
                time.sleep(delay)
                continue
            else:
                raise APICallError(
                    f"Gagal setelah {max_retries} percobaan -- API terus mengembalikan "
                    f"HTTP {response.status_code}.\n"
                    f"      Kalau ini kode 429 dan sudah terjadi berkali-kali walau kredit "
                    f"masih ada dan sudah ditunggu lama, KEMUNGKINAN BESAR ini masalah di "
                    f"sisi provider ({url}) -- bukan bug di kode kita. Coba:\n"
                    f"      1. Cek dashboard/Discord support provider untuk status akun.\n"
                    f"      2. Coba model lain di provider yang sama untuk isolasi masalah.\n"
                    f"      3. Kalau tetap gagal, pindahkan BASE_URL di config.py ke "
                    f"provider lain (OpenAI langsung / OpenRouter / dll).\n"
                    f"      Detail terakhir: {last_error_detail}"
                )

        # ---- Error PERMANEN (400, 401, 403, 404, dll) -> JANGAN retry ----
        if response.status_code != 200:
            print("STATUS:", response.status_code)
            print("BODY:", response.text)
            response.raise_for_status()

        return response.json()

    raise APICallError(last_error_detail or "Gagal memanggil API tanpa detail error.")