"""
evaluation/run_full_evaluation.py

Menjalankan evaluasi penelitian:

TAHAP 1 : Accuracy / Precision / Recall / F1
TAHAP 2 : Robustness
TAHAP 3 : Efisiensi + Realtime Frame + Ablation
TAHAP 4 : RAGAS

Perintah:

python test_ablation.py

kemudian:

python evaluation/run_full_evaluation.py --skip-ragas
"""

import asyncio
import sys

from live_evaluator import LiveEvaluator
from ablation_evaluator import AblationEvaluator
from realtime_frame_evaluator import RealtimeFrameEvaluator

import batch_ragas_eval


def jalankan_tahap_1_dan_2(ev):
    print("TAHAP 1: ACCURACY / PRECISION / RECALL / F1")
    ev.compute_batch_accuracy()

    print("TAHAP 2: ROBUSTNESS BREAKDOWN PER KONDISI PENGUJIAN")
    ev.compute_robustness_breakdown()


def jalankan_tahap_3a(ev):
    print("[TAHAP 3A] RINGKASAN EFISIENSI")
    ev.print_ringkasan_efisiensi()


def jalankan_tahap_3b():
    print("[TAHAP 3B] EVALUASI REALTIME FRAME")

    evaluator = RealtimeFrameEvaluator()
    hasil = evaluator.print_summary()

    if hasil is None:
        print("[INFO] Belum ada data frame realtime.")
        print("[INFO] Jalankan pengujian Live Camera terlebih dahulu.")


def jalankan_tahap_3c():
    print("[TAHAP 3C] ABLATION STUDY")

    evaluator = AblationEvaluator()
    hasil = evaluator.print_summary()

    if hasil is None:
        print("[INFO] Belum ada data Ablation Study.")
        print("[INFO] Jalankan terlebih dahulu:")
        print("       python test_ablation.py")


def jalankan_tahap_3(ev):
    print("TAHAP 3: EFISIENSI DAN ABLATION STUDY")

    jalankan_tahap_3a(ev)
    jalankan_tahap_3b()
    jalankan_tahap_3c()


async def jalankan_tahap_4():
    print("TAHAP 4: RAGAS - FAITHFULNESS & ANSWER RELEVANCY")
    await batch_ragas_eval.main()


def main():
    skip_ragas = "--skip-ragas" in sys.argv

    ev = LiveEvaluator()

    jalankan_tahap_1_dan_2(ev)
    jalankan_tahap_3(ev)

    if skip_ragas:
        print("[INFO] Tahap 4 (RAGAS) dilewati karena flag --skip-ragas.")
        return

    asyncio.run(jalankan_tahap_4())


if __name__ == "__main__":
    main()