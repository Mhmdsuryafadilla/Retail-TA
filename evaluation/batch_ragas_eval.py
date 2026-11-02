"""
evaluation/batch_ragas_eval.py

Script OFFLINE untuk menghitung Faithfulness & Answer Relevancy dari
data yang sudah di-log otomatis oleh live_evaluator.py ke evaluation_log.csv.
Dijalankan terpisah dari aplikasi utama karena metrik ini butuh LLM judge
tambahan yang mahal kalau dipanggil di setiap request produksi.

CARA PAKAI:
    python evaluation/batch_ragas_eval.py

Hasil:
    1. Dicetak ringkasan ke terminal
    2. Disimpan ke evaluation/ragas_batch_results.csv (per-baris)

PERSIAPAN:
    pip install ragas langchain-openai pandas --break-system-packages
    Isi bagian KONFIGURASI di bawah sesuai provider/router yang dipakai
    (sama dengan config.py).

CATATAN:
    - Butuh koneksi internet & API key aktif (LLM judge + embedding).
    - Kalau router tidak sediakan endpoint embeddings, Answer Relevancy
      akan gagal -- lihat alternatif embedding lokal di akhir file.
    - Baris dengan produk_ditemukan=False dilewati dari Faithfulness
      (tidak ada context untuk dicek), tapi tetap dicatat sebagai
      "N/A - produk tidak ditemukan".
"""

import asyncio
import os
import re
import sys
import time

import pandas as pd

# KONFIGURASI - sesuaikan dengan config.py project kamu
API_KEY = os.environ.get("OPENAI_API_KEY", "OPENAI_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "openai/gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"          # dipakai hanya untuk Answer Relevancy

CSV_LOG_PATH = os.path.join(os.path.dirname(__file__), "evaluation_detail.csv")
CSV_RESULT_PATH = os.path.join(os.path.dirname(__file__), "ragas_batch_results.csv")

# Fallback kalau nama produk gagal diekstrak dari context_text
PERTANYAAN_IMPLISIT_FALLBACK = (
    "Apa informasi, harga, dan rekomendasi yang berguna untuk konsumen "
    "terkait produk ini?"
)


def extract_nama_produk(context_text):
    """Ambil nama_produk dari blok PRODUK: saja, bukan dari blok PEMBANDING:."""
    if not isinstance(context_text, str):
        return None

    blok_match = re.search(r"PRODUK:(.*?)PEMBANDING:", context_text, re.DOTALL)
    teks_dicari = blok_match.group(1) if blok_match else context_text

    nama_match = re.search(r"nama_produk:\s*(.+)", teks_dicari)
    if nama_match:
        return nama_match.group(1).strip()

    return None


def build_pertanyaan_implisit(nama_produk):
    """Bikin pertanyaan implisit spesifik per produk agar Answer Relevancy akurat."""
    if nama_produk:
        return (
            f"Berapa harga {nama_produk} dan apakah produk ini worth "
            f"dibeli dibanding produk sejenis lainnya?"
        )
    return PERTANYAAN_IMPLISIT_FALLBACK


def jelaskan_error(e):
    """Gali detail error tambahan (status code, response body) dari exception."""
    detail = [f"Tipe error: {type(e).__name__}", f"Pesan: {e}"]

    response = getattr(e, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            detail.append(f"HTTP Status Code: {status_code}")
        try:
            body = response.text
            detail.append(f"Response body: {body[:500]}")
        except Exception:
            pass

    for attr in ("status_code", "code", "body"):
        val = getattr(e, attr, None)
        if val is not None:
            detail.append(f"{attr}: {str(val)[:500]}")

    return "\n      ".join(detail)


async def hitung_dengan_retry(scorer, sample, nama_metrik, max_retries=3, delay_detik=5):
    """Retry pemanggilan scorer karena error blocked/rate-limit sering sementara."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return await scorer.single_turn_ascore(sample), None
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"   [RETRY {attempt}/{max_retries}] {nama_metrik} gagal, "
                      f"menunggu {delay_detik} detik sebelum coba lagi...")
                await asyncio.sleep(delay_detik)
            else:
                print(f"   [ERROR] Gagal hitung {nama_metrik} setelah {max_retries} percobaan:")
                print(f"      {jelaskan_error(e)}")
    return None, last_error


def is_produk_ditemukan(value):
    """CSV menyimpan boolean sebagai teks 'True'/'False' -- baca dengan aman."""
    return str(value).strip().lower() == "true"


async def main():

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness, ResponseRelevancy
    except ImportError as e:
        print("[ERROR] Library belum lengkap ter-install.")
        print("Jalankan dulu: pip install ragas langchain-openai pandas --break-system-packages")
        print("Detail error:", e)
        sys.exit(1)

    if not os.path.exists(CSV_LOG_PATH):
        print(f"[ERROR] File {CSV_LOG_PATH} tidak ditemukan. "
              "Jalankan dulu aplikasi utama (app.py) beberapa kali supaya "
              "ada data ter-log, baru jalankan script ini.")
        sys.exit(1)

    df = pd.read_csv(CSV_LOG_PATH)

    # Validasi header vs data, untuk deteksi CSV lama yang kolomnya beda
    with open(CSV_LOG_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    if len(raw_lines) > 1:
        if "produk_ditemukan" not in df.columns:
            print("[ERROR] Kolom 'produk_ditemukan' tidak ditemukan di CSV.")
            print("        Kemungkinan struktur kolom evaluation_log.csv berbeda")
            print("        dari yang diharapkan (versi live_evaluator.py lama vs baru).")
            print("        SOLUSI: rename/hapus evaluation_log.csv yang lama,")
            print("        lalu jalankan ulang app.py untuk generate log baru.")
            sys.exit(1)

    if len(df) == 0:
        print("[INFO] evaluation_log.csv masih kosong, tidak ada yang dievaluasi.")
        sys.exit(0)

    jumlah_ditemukan = df["produk_ditemukan"].apply(is_produk_ditemukan).sum()
    if jumlah_ditemukan == 0:
        print("[WARNING] SEMUA baris di evaluation_log.csv menunjukkan")
        print("          produk_ditemukan=False. Cek apakah header CSV")
        print("          sinkron dengan jumlah kolom data.\n")

    # Setup LLM judge + embedding via router yang sama dengan llm_api.py.
    # default_headers dipaksa menyamai request polos (requests) agar
    # tidak diblokir WAF yang sensitif ke User-Agent SDK resmi.
    HEADERS_MENYAMAI_REQUESTS = {
        "User-Agent": "python-requests/2.31.0",
    }

    chat_model = ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        default_headers=HEADERS_MENYAMAI_REQUESTS,
    )
    evaluator_llm = LangchainLLMWrapper(chat_model)

    try:
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            default_headers=HEADERS_MENYAMAI_REQUESTS,
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
        embeddings_available = True
    except Exception as e:
        print("[WARNING] Gagal setup embeddings dari router ini.")
        print("          Answer Relevancy akan DILEWATI untuk semua baris.")
        print("          Detail:", e)
        print("          (Lihat 'ALTERNATIF EMBEDDING LOKAL' di akhir file)")
        evaluator_embeddings = None
        embeddings_available = False

    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    relevancy_scorer = (
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        if embeddings_available else None
    )

    results = []

    print(f"MEMULAI EVALUASI BATCH RAGAS — {len(df)} baris ditemukan")

    for idx, row in df.iterrows():

        no = idx + 1
        produk_ditemukan = is_produk_ditemukan(row.get("produk_ditemukan"))
        context_text = str(row.get("context_text", "") or "")
        answer_text = str(row.get("ai_answer_text", "") or "")
        barcode = row.get("barcode_dipakai", "-")

        print(f"\n[{no}/{len(df)}] Barcode: {barcode} | Produk ditemukan: {produk_ditemukan}")

        if not produk_ditemukan:
            print("   -> Dilewati (tidak ada context produk untuk dicek Faithfulness)")
            results.append({
                "no": no,
                "barcode": barcode,
                "produk_ditemukan": False,
                "faithfulness": "N/A - produk tidak ditemukan",
                "answer_relevancy": "N/A - produk tidak ditemukan",
            })
            continue

        nama_produk = extract_nama_produk(context_text)
        pertanyaan = build_pertanyaan_implisit(nama_produk)

        print(f"   Pertanyaan dipakai: \"{pertanyaan}\"")

        sample = SingleTurnSample(
            user_input=pertanyaan,
            response=answer_text,
            retrieved_contexts=[context_text],
        )

        faith_score, _ = await hitung_dengan_retry(
            faithfulness_scorer, sample, "Faithfulness"
        )

        rel_score = None
        if relevancy_scorer is not None:
            rel_score, _ = await hitung_dengan_retry(
                relevancy_scorer, sample, "Answer Relevancy"
            )

        faith_score_bulat = round(faith_score, 3) if isinstance(faith_score, (int, float)) else faith_score
        rel_score_bulat = round(rel_score, 3) if isinstance(rel_score, (int, float)) else rel_score

        print(f"   -> Faithfulness      : {faith_score_bulat}")
        print(f"   -> Answer Relevancy  : {rel_score_bulat}")

        results.append({
            "no": no,
            "barcode": barcode,
            "produk_ditemukan": True,
            "pertanyaan_dipakai": pertanyaan,
            "faithfulness": faith_score_bulat,
            "answer_relevancy": rel_score_bulat if rel_score_bulat is not None else "Tidak dihitung (embeddings tidak tersedia)",
        })

        # Jeda antar-baris agar tidak kena rate limit/WAF
        if idx < len(df) - 1:
            time.sleep(3)

    # Ringkasan
    valid_faith = [r["faithfulness"] for r in results if isinstance(r["faithfulness"], (int, float))]
    valid_rel = [r["answer_relevancy"] for r in results if isinstance(r["answer_relevancy"], (int, float))]

    baris_ringkasan = [
        ("Total baris", str(len(df))),
        ("Baris dengan produk ditemukan", str(sum(1 for r in results if r['produk_ditemukan']))),
        ("Baris dievaluasi Faithfulness", str(len(valid_faith))),
    ]
    if valid_faith:
        baris_ringkasan.append(("Rata-rata Faithfulness", f"{sum(valid_faith)/len(valid_faith):.3f}"))
    if valid_rel:
        baris_ringkasan.append(("Rata-rata Answer Relevancy", f"{sum(valid_rel)/len(valid_rel):.3f}"))
    elif embeddings_available is False:
        baris_ringkasan.append(("Rata-rata Answer Relevancy", "tidak dihitung (embeddings tidak tersedia)"))

    lebar_label = max(len(l) for l, _ in baris_ringkasan)

    print("RINGKASAN HASIL EVALUASI BATCH RAGAS")
    for label, nilai in baris_ringkasan:
        print(f"{label.ljust(lebar_label)} : {nilai}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(CSV_RESULT_PATH, index=False)
    print(f"\nHasil detail per-baris disimpan ke: {CSV_RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
