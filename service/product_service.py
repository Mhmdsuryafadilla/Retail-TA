# service/product_service.py

import time
import os
DATASET_CLEAN_DIR = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "dataset_images"
)
import cv2

from vision.vision_api import VisionAPI
from vision.barcode_detector import detect_barcode
from vision.preprocessing import preprocess_image, crop_to_bbox, scale_bbox
from vision.barcode_validator import validate_checksum

from database.product_repository import ProductRepository

from llm.llm_api import LLMAPI
from prompt.prompt_template import PromptTemplate

from evaluation.live_evaluator import LiveEvaluator
from evaluation.scan_archiver import save_scan_image
from evaluation.condition_detector import detect_semua_kondisi, print_ringkas_kondisi

from decimal import Decimal


class ProductService:

    # Resolusi tinggi untuk pyzbar, resolusi rendah untuk Vision AI (hemat token)
    PYZBAR_MAX_WIDTH = 1600
    VISION_MAX_WIDTH = 800

    def __init__(self):
        self.vision = VisionAPI()
        self.repo = ProductRepository()
        self.llm = LLMAPI()
        self.evaluator = LiveEvaluator()

    # UTILITAS

    def safe_number(self, value):
        try:
            if value is None:
                return 0
            if isinstance(value, Decimal):
                return float(value)
            return float(value)
        except Exception:
            return 0

    def _clean_value(self, value):
        if isinstance(value, Decimal):
            return float(value)
        return value

    def _dict_to_text(self, data, indent=0):
        pad = "  " * indent

        if data is None:
            return f"{pad}(tidak ada data)"

        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                value = self._clean_value(value)
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.append(self._dict_to_text(value, indent + 1))
                else:
                    lines.append(f"{pad}{key}: {value}")
            return "\n".join(lines)

        if isinstance(data, list):
            if not data:
                return f"{pad}(kosong)"
            lines = []
            for i, item in enumerate(data, start=1):
                lines.append(f"{pad}{i}.")
                lines.append(self._dict_to_text(item, indent + 1))
            return "\n".join(lines)

        return f"{pad}{self._clean_value(data)}"

    @staticmethod
    def _cleanup_temp_file(path):
        """Hapus file temp, aman jika gagal (bukan kritikal)."""
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[DEBUG] Gagal menghapus file sementara '{path}': {e}")

    # ANALISIS HARGA

    def calculate_price_analysis(self, product, similar_products):
        analysis = {}

        harga = self.safe_number(product.get("harga"))
        qty = self.safe_number(product.get("qty_val"))

        if qty > 0:
            analysis["harga_satuan"] = round(harga / qty, 2)
        else:
            analysis["harga_satuan"] = "Tidak tersedia"

        produk_jenis = product.get("jenis_spesifik")
        produk_satuan = product.get("satuan")

        similar_products_valid = []
        for item in similar_products:
            item_jenis = item.get("jenis_spesifik")
            item_satuan = item.get("satuan")

            if produk_jenis and item_jenis and item_jenis != produk_jenis:
                continue
            if produk_satuan and item_satuan and item_satuan != produk_satuan:
                continue

            similar_products_valid.append(item)

        similar_products = similar_products_valid

        produk_termurah = None
        harga_termurah = None

        ukuran_terbesar = None
        ukuran_max = 0

        for item in similar_products:
            item_harga = self.safe_number(item.get("harga"))
            item_qty = self.safe_number(item.get("qty_val"))

            if item_harga > 0:
                if harga_termurah is None or item_harga < harga_termurah:
                    harga_termurah = item_harga
                    produk_termurah = item

            if item_qty > ukuran_max:
                ukuran_max = item_qty
                ukuran_terbesar = item

        if produk_termurah:
            analysis["produk_termurah"] = produk_termurah

        if ukuran_terbesar:
            analysis["ukuran_terbesar"] = ukuran_terbesar

        analysis["alternatif"] = similar_products[:3]

        return analysis

    # PROSES ABLATION STUDY

    def process_ablation(self, image_path, barcode_asli, konfigurasi):
        """Jalankan satu gambar untuk satu konfigurasi ablation (A/B/C/D)."""
        konfigurasi = konfigurasi.strip()

        allowed = ["A-Pyzbar", "B-LVLM", "C-Hybrid", "D-Sistem Lengkap"]
        if konfigurasi not in allowed:
            raise ValueError(f"Konfigurasi tidak dikenal: {konfigurasi}")

        t_total = time.perf_counter()

        frame_asli = cv2.imread(image_path)
        if frame_asli is None:
            raise ValueError(f"Gagal membaca gambar: {image_path}")

        t_pre = time.perf_counter()
        frame_proses = preprocess_image(frame_asli, max_width=800)
        latency_preprocessing = time.perf_counter() - t_pre

        barcode_pyzbar = None
        barcode_vision = None
        barcode_final = None

        found_pyzbar = False
        bbox = None

        latency_pyzbar = 0.0
        latency_crop = 0.0
        latency_vision = 0.0
        latency_database = 0.0
        latency_llm = 0.0

        token_vision = 0
        token_llm = 0

        product = None
        answer = ""

        vision_temp_path = None

        try:
            # A -- Pyzbar
            t0 = time.perf_counter()
            found_pyzbar_kondisi, barcode_pyzbar_kondisi, bbox_kondisi = detect_barcode(frame_proses)
            latency_deteksi_kondisi = time.perf_counter() - t0

            checksum_valid_kondisi = (
                validate_checksum(barcode_pyzbar_kondisi)
                if found_pyzbar_kondisi and barcode_pyzbar_kondisi
                else False
            )

            kondisi_otomatis, kondisi_debug = detect_semua_kondisi(
                frame_proses,
                bbox_kondisi if found_pyzbar_kondisi else None,
                found_pyzbar_kondisi,
                checksum_valid_kondisi,
            )
            image_detail_level = kondisi_otomatis["image_detail_level"]

            if konfigurasi in ["A-Pyzbar", "C-Hybrid", "D-Sistem Lengkap"]:
                found_pyzbar = found_pyzbar_kondisi
                barcode_pyzbar = barcode_pyzbar_kondisi if found_pyzbar_kondisi else None
                bbox = bbox_kondisi
                latency_pyzbar = latency_deteksi_kondisi

            # B/C/D -- Vision AI
            perlu_vision = konfigurasi in ["B-LVLM", "C-Hybrid", "D-Sistem Lengkap"]
            frame_vision = frame_proses

            if konfigurasi in ["C-Hybrid", "D-Sistem Lengkap"]:
                if found_pyzbar and bbox is not None:
                    t_crop = time.perf_counter()
                    frame_vision = crop_to_bbox(frame_proses, bbox, padding=60)
                    latency_crop = time.perf_counter() - t_crop

            if perlu_vision:
                vision_temp_path = "temp_ablation_vision.jpg"
                cv2.imwrite(vision_temp_path, frame_vision)

                t0 = time.perf_counter()
                vision_result = self.vision.identify_product(
                    vision_temp_path,
                    image_detail_level=image_detail_level,
                )
                latency_vision = time.perf_counter() - t0

                barcode_raw = vision_result.get("barcode") or vision_result.get("gambar_barcode", "")
                if barcode_raw:
                    barcode_vision = str(barcode_raw).strip()

                usage = self.vision.last_usage or {}
                token_vision = int(usage.get("total_tokens", 0) or 0)

            # Tentukan barcode konfigurasi
            if konfigurasi == "A-Pyzbar":
                barcode_final = barcode_pyzbar
            elif konfigurasi == "B-LVLM":
                barcode_final = barcode_vision
            elif konfigurasi in ["C-Hybrid", "D-Sistem Lengkap"]:
                if barcode_pyzbar:
                    barcode_final = barcode_pyzbar
                elif barcode_vision:
                    barcode_final = barcode_vision

            # D -- Database
            if konfigurasi == "D-Sistem Lengkap":
                if barcode_final:
                    t0 = time.perf_counter()
                    product = self.repo.get_by_barcode(barcode_final)
                    latency_database = time.perf_counter() - t0

            # D -- AI Explainer
            if konfigurasi == "D-Sistem Lengkap":
                similar_products = []
                analysis = {}
                total_valid_di_db = None

                if product:
                    similar_products = self.repo.get_similar_products(product)
                    total_valid_di_db = self.repo.get_similar_products_count(product)
                    analysis = self.calculate_price_analysis(product, similar_products)

                product_text = self._dict_to_text(product)
                comparison_text = self._dict_to_text(similar_products)
                analysis_text = self._dict_to_text(analysis)

                if product:
                    prompt = PromptTemplate.product_explanation().format(
                        product_data=product_text,
                        comparison_data=comparison_text,
                        analysis=analysis_text
                    )
                else:
                    prompt = PromptTemplate.product_not_found().format(
                        product_name=barcode_final or "(tidak terbaca)"
                    )

                t0 = time.perf_counter()
                answer = self.llm.generate(prompt)
                latency_llm = time.perf_counter() - t0

                if self.llm.last_usage:
                    token_llm = int(self.llm.last_usage.get("total_tokens", 0) or 0)

        finally:
            # Khusus ablation: selalu dibersihkan, tidak pernah diarsipkan
            self._cleanup_temp_file(vision_temp_path)

        latency_total = time.perf_counter() - t_total

        pred = str(barcode_final).strip() if barcode_final else ""
        actual = str(barcode_asli).strip() if barcode_asli else ""
        benar = bool(pred) and bool(actual) and pred == actual

        cer = self._calculate_cer(pred, actual)

        database_match = None
        if konfigurasi == "D-Sistem Lengkap":
            database_match = product is not None

        return {
            "konfigurasi": konfigurasi,
            "image_path": image_path,
            "barcode_asli": actual,
            "barcode_pyzbar": barcode_pyzbar or "",
            "barcode_vision": barcode_vision or "",
            "barcode_final": barcode_final or "",
            "benar": benar,
            "cer": cer,
            "latency_preprocessing": latency_preprocessing,
            "latency_deteksi_kondisi": latency_deteksi_kondisi,
            "latency_pyzbar": latency_pyzbar,
            "latency_crop": latency_crop,
            "latency_vision": latency_vision,
            "latency_database": latency_database,
            "latency_llm": latency_llm,
            "latency_total": latency_total,
            "token_vision": token_vision,
            "token_llm": token_llm,
            "token_total_ai": token_vision + token_llm,
            "database_match": database_match,
            "produk_ditemukan": product is not None,
            "kondisi_pencahayaan": kondisi_otomatis["kondisi_pencahayaan"],
            "kondisi_fisik_barcode": kondisi_otomatis["kondisi_fisik_barcode"],
            "image_detail_level": image_detail_level,
        }

    # CER

    @staticmethod
    def _calculate_cer(prediksi, aktual):
        if not aktual:
            return None
        if not prediksi:
            return 1.0

        a = str(prediksi)
        b = str(aktual)

        if a == b:
            return 0.0

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i]
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr.append(min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost
                ))
            prev = curr

        return round(prev[-1] / max(len(b), 1), 3)

    # PEMBACAAN BARCODE SISTEM UTAMA

    def _baca_barcode(self, image_path):
        """
        preprocessing -> pyzbar (hires) -> deteksi kondisi -> crop (res vision)
        -> Vision AI -> agreement -> checksum -> kondisi final.
        Pyzbar & Vision AI jalan di resolusi berbeda dari frame_asli yang sama,
        jadi bbox pyzbar di-scale dulu ke skala frame vision sebelum dipakai.
        """
        frame_asli = cv2.imread(image_path)
        if frame_asli is None:
            raise ValueError(f"Gagal membaca gambar: {image_path}")

        frame_pyzbar = preprocess_image(frame_asli, max_width=self.PYZBAR_MAX_WIDTH)
        frame_proses = preprocess_image(frame_asli, max_width=self.VISION_MAX_WIDTH)

        t0 = time.perf_counter()
        found_pyzbar, barcode_pyzbar, bbox_hires = detect_barcode(frame_pyzbar)
        latency_pyzbar = round(time.perf_counter() - t0, 4)

        if not found_pyzbar:
            barcode_pyzbar = None
            bbox_hires = None

        pyzbar_checksum_valid = validate_checksum(barcode_pyzbar) if barcode_pyzbar else False

        bbox = None
        if found_pyzbar and bbox_hires is not None:
            bbox = scale_bbox(
                bbox_hires,
                from_width=frame_pyzbar.shape[1],
                to_width=frame_proses.shape[1],
            )

        gambar_di_crop = bool(found_pyzbar and bbox is not None)

        if gambar_di_crop:
            frame_untuk_vision = crop_to_bbox(frame_proses, bbox, padding=60)
        else:
            frame_untuk_vision = frame_proses

        vision_temp_path = "temp_vision_input.jpg"
        cv2.imwrite(vision_temp_path, frame_untuk_vision)

        bbox_untuk_deteksi = bbox if found_pyzbar and bbox is not None else None

        kondisi_otomatis, kondisi_debug = detect_semua_kondisi(
            frame_proses,
            bbox_untuk_deteksi,
            found_pyzbar,
            pyzbar_checksum_valid,
        )

        print_ringkas_kondisi(kondisi_otomatis, kondisi_debug)

        try:
            t0 = time.perf_counter()
            vision_result = self.vision.identify_product(
                vision_temp_path,
                image_detail_level=kondisi_otomatis["image_detail_level"],
            )
            latency_vision = round(time.perf_counter() - t0, 2)
        finally:
            self._cleanup_temp_file(vision_temp_path)

        barcode_vision_raw = vision_result.get("barcode") or vision_result.get("gambar_barcode", "")
        barcode_vision = str(barcode_vision_raw).strip() if barcode_vision_raw else None

        if barcode_pyzbar and barcode_vision:
            agreement = "Sama" if barcode_pyzbar == barcode_vision else "Beda"
        else:
            agreement = "Tidak dapat dibandingkan"

        vision_checksum_valid = validate_checksum(barcode_vision) if barcode_vision else False

        if barcode_pyzbar:
            barcode_final = barcode_pyzbar
            sumber_barcode = "pyzbar (checksum tervalidasi otomatis)"
        elif barcode_vision and vision_checksum_valid:
            barcode_final = barcode_vision
            sumber_barcode = "vision_ai (checksum tervalidasi manual)"
        elif barcode_vision:
            barcode_final = barcode_vision
            sumber_barcode = "vision_ai (PERINGATAN: checksum tidak valid)"
        else:
            barcode_final = None
            sumber_barcode = "tidak ada barcode terbaca dari kedua metode"

        vision_result["barcode"] = barcode_final
        vision_result["sumber_barcode"] = sumber_barcode
        vision_result["barcode_pyzbar"] = barcode_pyzbar
        vision_result["barcode_vision_asli"] = barcode_vision
        vision_result["gambar_di_crop_ke_area_barcode"] = gambar_di_crop
        vision_result["checksum_valid"] = (
            pyzbar_checksum_valid if barcode_pyzbar else vision_checksum_valid
        )
        vision_result["kondisi_pencahayaan_auto"] = kondisi_otomatis["kondisi_pencahayaan"]
        vision_result["kondisi_fisik_barcode_auto"] = kondisi_otomatis["kondisi_fisik_barcode"]
        vision_result["image_detail_level_auto"] = kondisi_otomatis["image_detail_level"]

        return {
            "barcode_final": barcode_final,
            "vision_result": vision_result,
            "barcode_pyzbar": barcode_pyzbar,
            "barcode_vision": barcode_vision,
            "pyzbar_checksum_valid": pyzbar_checksum_valid,
            "vision_checksum_valid": vision_checksum_valid,
            "agreement": agreement,
            "latency_vision": latency_vision,
            "latency_pyzbar": latency_pyzbar,
            "kondisi_otomatis": kondisi_otomatis,
            "kondisi_debug": kondisi_debug,
        }

    # PROCESS IMAGE - JALUR UTAMA APP.PY

    def process_image(self, image_path, mode="Tidak diketahui", barcode_asli=None, archive_image_path=None):
        """
        image_path         : gambar bersih, dipakai untuk pyzbar & Vision AI,
                              dihapus di akhir fungsi (tidak pernah diarsipkan).
        archive_image_path : opsional, gambar beranotasi. Hanya ini yang
                              disimpan permanen ke scan_images/.
        """

        baca = self._baca_barcode(image_path)
        barcode = baca["barcode_final"]
        vision_result = baca["vision_result"]

        if archive_image_path:
            save_scan_image(archive_image_path, barcode_hint=barcode, mode=mode)

        self._save_clean_dataset_copy(image_path, barcode, mode)

        print("[DEBUG] Barcode pyzbar :", baca["barcode_pyzbar"])
        print("[DEBUG] Barcode vision :", baca["barcode_vision"])
        print("[DEBUG] Agreement      :", baca["agreement"])
        print("[DEBUG] Barcode final untuk lookup:", barcode)

        product = None
        if barcode:
            product = self.repo.get_by_barcode(barcode)

        print("[DEBUG] Database:", product)

        if barcode_asli is None and product:
            barcode_asli = str(product.get("barcode"))
            print(
                f"[DEBUG] barcode_asli di-fallback dari database: {barcode_asli} "
                "(BUKAN ground truth fisik independen)"
            )

        similar_products = []
        analysis = {}
        total_valid_di_db = None

        if product:
            similar_products = self.repo.get_similar_products(product)
            total_valid_di_db = self.repo.get_similar_products_count(product)
            analysis = self.calculate_price_analysis(product, similar_products)

        product_text = self._dict_to_text(product)
        comparison_text = self._dict_to_text(similar_products)
        analysis_text = self._dict_to_text(analysis)

        if product:
            prompt = PromptTemplate.product_explanation().format(
                product_data=product_text,
                comparison_data=comparison_text,
                analysis=analysis_text
            )
        else:
            prompt = PromptTemplate.product_not_found().format(
                product_name=barcode or "(tidak terbaca)"
            )

        t1 = time.time()
        answer = self.llm.generate(prompt)
        latency_llm = round(time.time() - t1, 2)

        token_llm = None
        if self.llm.last_usage:
            token_llm = self.llm.last_usage.get("total_tokens")

        context_gabungan = (
            f"PRODUK:\n{product_text}\n\n"
            f"PEMBANDING:\n{comparison_text}\n\n"
            f"ANALISIS:\n{analysis_text}"
        )

        kondisi_otomatis = baca["kondisi_otomatis"]

        self.evaluator.evaluate_and_report(
            mode=mode,
            barcode_pyzbar=baca["barcode_pyzbar"],
            barcode_vision_tebakan=baca["barcode_vision"],
            barcode_dipakai=barcode,
            barcode_asli=barcode_asli,
            kondisi_pencahayaan=kondisi_otomatis["kondisi_pencahayaan"],
            kondisi_fisik_barcode=kondisi_otomatis["kondisi_fisik_barcode"],
            image_detail_level=kondisi_otomatis["image_detail_level"],
            kondisi_debug=baca["kondisi_debug"],
            pyzbar_checksum_valid=baca["pyzbar_checksum_valid"],
            vision_checksum_valid=baca["vision_checksum_valid"],
            gambar_di_crop=vision_result["gambar_di_crop_ke_area_barcode"],
            latency_pyzbar_detik=baca["latency_pyzbar"],
            product=product,
            similar_products=similar_products,
            total_valid_di_db=total_valid_di_db,
            token_llm_explainer=token_llm,
            latency_vision_detik=baca["latency_vision"],
            latency_llm_detik=latency_llm,
            context_text=context_gabungan,
            ai_answer_text=answer,
        )

        result = {
            "vision": vision_result,
            "product": product,
            "answer": answer
        }

        self._cleanup_temp_file(image_path)
        self._cleanup_temp_file(archive_image_path)

        return result

    def _save_clean_dataset_copy(self, image_path, barcode_hint, mode):
        """Simpan salinan permanen gambar bersih ke folder dataset ablation."""
        os.makedirs(DATASET_CLEAN_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_slug = mode.lower().replace(" ", "_")
        barcode_slug = barcode_hint if barcode_hint else "unknown"
        filename = f"{timestamp}_{mode_slug}_{barcode_slug}.jpg"
        dest_path = os.path.join(DATASET_CLEAN_DIR, filename)

        img = cv2.imread(image_path)
        if img is not None:
            cv2.imwrite(dest_path, img)
        return dest_path