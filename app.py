# app.py

import time
import hashlib
import threading
from datetime import datetime

import av
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from vision.vision_api import VisionAPI
from vision.barcode_detector import detect_barcode, draw_barcode_box
from service.product_service import ProductService

from evaluation.realtime_frame_evaluator import (
    RealtimeFrameEvaluator
)


# Init service
if "vision_service" not in st.session_state:
    st.session_state.vision_service = VisionAPI()

if "product_service" not in st.session_state:
    st.session_state.product_service = ProductService()

if "realtime_evaluator" not in st.session_state:
    st.session_state.realtime_evaluator = (
        RealtimeFrameEvaluator()
    )


vision_service = st.session_state.vision_service
product_service = st.session_state.product_service
realtime_evaluator = st.session_state.realtime_evaluator


# Config UI
st.set_page_config(
    page_title="Retail AI Vision System",
    layout="wide"
)

st.title(":material/shopping_cart: Retail AI Vision System")

st.write(
    "Uji sistem Vision → DB → LLM lewat upload gambar atau live camera"
)

st.caption(
    ":material/science: Kondisi pengujian (pencahayaan, fisik barcode, image detail level) "
    "dideteksi OTOMATIS dari tiap gambar yang di-scan -- tidak ada input "
    "manual. Lihat detail angka deteksinya di terminal tempat "
    "`streamlit run app.py` dijalankan."
)


# Helper: tampilkan hasil
def show_result(result, container=None):

    target = container if container is not None else st

    target.subheader(":material/search: Vision Result")
    target.json(result["vision"])

    target.subheader(":material/inventory_2: Database Result")

    if result["product"]:
        target.json(dict(result["product"]))
    else:
        target.warning("Data tidak ditemukan di database")

    target.subheader(":material/smart_toy: AI Answer")
    target.success(result["answer"])


# Tab upload vs live camera
tab_upload, tab_live = st.tabs(
    [":material/upload_file: Upload Gambar", ":material/photo_camera: Live Camera"]
)


# Tab 1: upload gambar
with tab_upload:

    uploaded_file = st.file_uploader(
        "Upload Gambar Produk",
        type=["jpg", "jpeg", "png"]
    )

    # State upload
    if "upload_result" not in st.session_state:
        st.session_state.upload_result = None

    if "upload_file_hash" not in st.session_state:
        st.session_state.upload_file_hash = None

    if "upload_barcode_caption" not in st.session_state:
        st.session_state.upload_barcode_caption = None


    if uploaded_file is not None:

        # Hash file
        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()


        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.image(
            image,
            caption="Input Image",
            width="stretch"
        )


        is_new_file = (
            file_hash
            != st.session_state.upload_file_hash
        )


        if is_new_file:

            # Gambar baru
            temp_path = "temp.jpg"

            image.save(temp_path)


            with st.spinner(
                "Memproses Vision AI..."
            ):

                result = product_service.process_image(
                    temp_path,
                    mode="Upload Gambar"
                )


            vision_info = result.get(
                "vision",
                {}
            )


            barcode_pyzbar = vision_info.get(
                "barcode_pyzbar"
            )


            barcode_vision = vision_info.get(
                "barcode_vision_asli"
            )


            if barcode_pyzbar:

                caption_text = (
                    f":material/photo_camera: Barcode terdeteksi (pyzbar): "
                    f"`{barcode_pyzbar}`"
                )

            elif barcode_vision:

                caption_text = (
                    f":material/photo_camera: Barcode terdeteksi (Vision AI): "
                    f"`{barcode_vision}`"
                )

            else:

                caption_text = (
                    "Tidak ada barcode yang berhasil terbaca "
                    "oleh pyzbar maupun Vision AI."
                )


            st.session_state.upload_result = result

            st.session_state.upload_file_hash = (
                file_hash
            )

            st.session_state.upload_barcode_caption = (
                caption_text
            )


        # Tampilkan hasil
        st.caption(
            st.session_state.upload_barcode_caption
        )


        show_result(
            st.session_state.upload_result
        )


    else:

        # Reset upload
        st.session_state.upload_result = None

        st.session_state.upload_file_hash = None

        st.session_state.upload_barcode_caption = None


        st.info(
            "Silakan upload gambar untuk mulai testing sistem."
        )


# Tab 2: live camera
with tab_live:

    # Pilih mode
    mode = st.radio(
        "Pilih Mode Capture",
        [
            "Mode Otomatis (auto-capture saat barcode terdeteksi)",
            "Mode Manual (tombol capture)"
        ],
        horizontal=True,
    )


    auto_mode = mode.startswith(
        "Mode Otomatis"
    )


    st.caption(
        "Tiap frame dicek pyzbar (deteksi barcode lokal & cepat, "
        "bukan LVLM). Begitu barcode terdeteksi, frame itu baru "
        "dikirim ke Vision AI untuk identifikasi produk + "
        "dicocokkan ke database."
    )


    # Shared state antar thread
    class SharedState:

        def __init__(self):

            self.lock = threading.Lock()

            # Frame
            self.latest_frame = None

            # Barcode
            self.barcode_detected = None

            self.barcode_frame = None

            # Frame beranotasi (ADA kotak hijau) -- HANYA untuk arsip
            # dokumentasi, TIDAK PERNAH dipakai untuk pyzbar/Vision AI.
            self.barcode_frame_annotated = None

            # Single trigger
            self.last_capture_time = 0

            self.locked = False

            # Real-time frame evaluation

            # Nomor frame dalam satu sesi scan
            self.frame_count = 0

            # Waktu awal sesi menggunakan perf_counter
            self.scan_start_time = None

            # Waktu awal sesi dalam format tanggal/waktu
            self.scan_start_datetime = None

            # Frame ketika barcode pertama ditemukan
            self.detection_frame = None

            # Waktu ketika barcode pertama ditemukan
            self.detection_time = None

            # Frame yang digunakan untuk proses
            self.processed_frame = None

            # Waktu ketika frame mulai diproses
            self.processed_time = None

            # Lama proses Vision AI
            self.total_processing_time = None

            # FPS aktual
            self.fps = 0.0

            # Counter FPS
            self.fps_frame_count = 0

            # Waktu awal perhitungan FPS
            self.fps_start_time = time.perf_counter()

            # Detail setiap frame
            self.frame_records = []

            # Evaluation
            self.scan_id = None

            self.evaluation_saved = False


    # Buat shared state
    if "shared_state" not in st.session_state:

        st.session_state.shared_state = SharedState()


    shared = st.session_state.shared_state


    # Session state live
    if "live_result" not in st.session_state:
        st.session_state.live_result = None

    if "live_error" not in st.session_state:
        st.session_state.live_error = None

    if "live_barcode" not in st.session_state:
        st.session_state.live_barcode = None


    # Callback ini dipanggil untuk setiap frame yang diterima oleh streamlit-webrtc
    def video_frame_callback(
        frame: av.VideoFrame
    ) -> av.VideoFrame:

        # Konversi frame
        img = frame.to_ndarray(
            format="bgr24"
        )

        # Cek lock sebelum scanning: jika barcode sudah ditemukan,
        # frame tidak dihitung lagi (tidak ada pyzbar/FPS/trigger baru),
        # frame hanya diteruskan ke tampilan kamera
        with shared.lock:

            if shared.locked:

                return av.VideoFrame.from_ndarray(
                    img,
                    format="bgr24"
                )


        # Waktu frame diterima
        current_time = time.perf_counter()


        # Mulai sesi scan
        with shared.lock:

            if shared.scan_start_time is None:

                shared.scan_start_time = current_time

                shared.scan_start_datetime = (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                shared.frame_count = 0

                shared.frame_records = []

                shared.scan_id = (
                    realtime_evaluator.get_next_scan_id()
                )

                shared.evaluation_saved = False

                print(
                    "\n"
                    "========================================\n"
                    "SCAN SESSION DIMULAI\n"
                    "========================================"
                )

                print(
                    f"Scan ID           : "
                    f"{shared.scan_id}"
                )

                print(
                    f"Start time        : "
                    f"{shared.scan_start_datetime}"
                )


            # Frame counter
            shared.frame_count += 1

            current_frame_number = (
                shared.frame_count
            )


            # Waktu relatif frame
            frame_elapsed = (
                current_time
                - shared.scan_start_time
            )


            # FPS aktual
            shared.fps_frame_count += 1

            fps_elapsed = (
                current_time
                - shared.fps_start_time
            )


            if fps_elapsed >= 1.0:

                shared.fps = (
                    shared.fps_frame_count
                    / fps_elapsed
                )

                print(
                    f"[FPS] "
                    f"{shared.fps:.2f} frame/detik | "
                    f"{shared.fps_frame_count} frame | "
                    f"{fps_elapsed:.3f} detik"
                )

                shared.fps_frame_count = 0

                shared.fps_start_time = (
                    current_time
                )


        # Deteksi barcode
        found, barcode_value, bbox = (
            detect_barcode(img)
        )

        # Simpan frame BERSIH di sini -- SEBELUM ada gambar apa pun (kotak/teks)
        # ditimpakan ke pixel-nya. Frame inilah yang nanti dipakai untuk proses
        # ulang (pyzbar kedua kalinya di product_service, + Vision AI), supaya
        # tidak tercemar anotasi visual yang kita gambar untuk tampilan kamera.
        clean_frame = img.copy()


        # Simpan frame terbaru
        with shared.lock:

            shared.latest_frame = clean_frame.copy()

            shared.frame_records.append({
                "frame_number": current_frame_number,
                "elapsed_time": frame_elapsed,
                "barcode_detected": bool(found)
            })

            # Barcode ditemukan
            if found:

                img = draw_barcode_box(
                    img,
                    bbox,
                    label=f"Barcode: {barcode_value}"
                )
                # img (dengan kotak) HANYA dipakai untuk tampilan live camera
                # di bawah dan untuk arsip dokumentasi -- bukan untuk
                # barcode_frame yang diproses ulang.

                now = time.time()

                if (now - shared.last_capture_time > 3):

                    shared.locked = True
                    shared.barcode_detected = barcode_value

                    # Frame BERSIH -- dipakai untuk proses (pyzbar/Vision AI)
                    shared.barcode_frame = clean_frame.copy()

                    # Frame BERANOTASI -- hanya untuk arsip dokumentasi
                    shared.barcode_frame_annotated = img.copy()

                    shared.detection_frame = current_frame_number
                    shared.detection_time = frame_elapsed
                    shared.last_capture_time = now

                    if shared.frame_records:
                        shared.frame_records[-1]["barcode_detected"] = True

                    if frame_elapsed > 0:
                        calculated_fps = current_frame_number / frame_elapsed
                        if shared.fps <= 0:
                            shared.fps = calculated_fps


                    # Output terminal
                    print(
                        "\n"
                        "========================================\n"
                        "BARCODE TERDETEKSI\n"
                        "========================================\n"
                        f"Scan ID           : "
                        f"{shared.scan_id}\n"
                        f"Barcode           : "
                        f"{barcode_value}\n"
                        f"Frame deteksi     : "
                        f"{current_frame_number}\n"
                        f"Waktu deteksi     : "
                        f"{frame_elapsed:.3f} detik\n"
                        f"FPS aktual        : "
                        f"{shared.fps:.2f} FPS\n"
                        f"Frames diperiksa  : "
                        f"{len(shared.frame_records)}\n"
                        "========================================\n"
                        "SCAN DIKUNCI - MENUNGGU PROSES SELESAI\n"
                        "========================================"
                    )


        # Kembalikan frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


    # Live camera stream
    col_cam, col_result = st.columns(
        [1, 1]
    )


    with col_cam:

        st.subheader("Live Camera")


        webrtc_ctx = webrtc_streamer(

            key="retail-scanner",

            mode=WebRtcMode.SENDRECV,

            video_frame_callback=(
                video_frame_callback
            ),

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True,
        )


        if not auto_mode:

            capture_clicked = st.button(
                "Capture Sekarang",
                icon=":material/photo_camera:",
                width="stretch"
            )

        else:

            capture_clicked = False

            st.info(
                "Mode otomatis aktif — arahkan kamera "
                "ke barcode produk, sistem akan capture sendiri."
            )


    # Proses hasil capture manual
    def process_frame_and_show(
        frame_bgr,
        container,
        archive_frame_bgr=None
    ):
        """
        frame_bgr          : frame BERSIH -- dipakai untuk diproses.
        archive_frame_bgr  : opsional, frame BERANOTASI (ada kotak hijau)
                              -- hanya untuk arsip dokumentasi.
        """

        temp_path = "temp_live.jpg"

        cv2.imwrite(
            temp_path,
            frame_bgr
        )

        # Simpan frame beranotasi (kalau ada) untuk arsip
        archive_path = None

        if archive_frame_bgr is not None:

            archive_path = "temp_live_annotated.jpg"

            cv2.imwrite(
                archive_path,
                archive_frame_bgr
            )


        with container:

            with st.spinner(
                "Memproses Vision AI..."
            ):

                result = (
                    product_service.process_image(
                        temp_path,
                        mode="Live Camera Manual",
                        archive_image_path=archive_path,
                    )
                )


            show_result(
                result,
                container
            )


    # Simpan evaluasi realtime
    def save_realtime_evaluation():

        # Cek agar tidak disimpan dua kali
        with shared.lock:

            if shared.evaluation_saved:

                return

            if shared.scan_id is None:

                return

            if shared.scan_start_time is None:

                return


            # Waktu akhir sesi
            end_time = time.perf_counter()

            elapsed_time = (
                end_time
                - shared.scan_start_time
            )


            # FPS final
            if elapsed_time > 0:

                fps_final = (
                    shared.frame_count
                    / elapsed_time
                )

            else:

                fps_final = shared.fps


            # Data session
            scan_id = shared.scan_id

            start_time = (
                shared.scan_start_datetime
            )

            frame_number = (
                shared.frame_count
            )

            barcode_detected = (
                shared.barcode_detected
                is not None
            )

            detection_frame = (
                shared.detection_frame
            )

            detection_time = (
                shared.detection_time
            )

            processed_frame = (
                shared.processed_frame
            )

            processed_time = (
                shared.processed_time
            )

            total_processing_time = (
                shared.total_processing_time
            )

            frames_checked = (
                len(shared.frame_records)
            )

            frame_records = list(
                shared.frame_records
            )


        # Simpan ke CSV
        try:

            realtime_evaluator.save_session(

                scan_id=scan_id,

                start_time=start_time,

                elapsed_time=elapsed_time,

                frame_number=frame_number,

                barcode_detected=barcode_detected,

                detection_frame=detection_frame,

                detection_time=detection_time,

                processed_frame=processed_frame,

                processed_time=processed_time,

                fps_actual=fps_final,

                frames_checked=frames_checked,

                total_processing_time=(
                    total_processing_time
                ),

                frame_records=frame_records
            )


            # Tandai sudah disimpan
            with shared.lock:

                shared.evaluation_saved = True


            # Output terminal
            print(
                "\n"
                "========================================\n"
                "EVALUASI REALTIME TERSIMPAN\n"
                "========================================\n"
                f"Scan ID              : "
                f"{scan_id}\n"
                f"Start time           : "
                f"{start_time}\n"
                f"Total frame          : "
                f"{frame_number}\n"
                f"Barcode detected     : "
                f"{barcode_detected}\n"
                f"Detection frame      : "
                f"{detection_frame}\n"
                f"Detection time       : "
                f"{detection_time}\n"
                f"Processed frame      : "
                f"{processed_frame}\n"
                f"Processed time       : "
                f"{processed_time}\n"
                f"FPS actual           : "
                f"{fps_final:.3f}\n"
                f"Frames checked       : "
                f"{frames_checked}\n"
                f"Processing time      : "
                f"{total_processing_time}\n"
                f"Elapsed time         : "
                f"{elapsed_time:.3f}\n"
                "========================================\n"
                "CSV SUMMARY           : "
                "realtime_scan_summary.csv\n"
                "CSV FRAME DETAIL      : "
                "realtime_frame_log.csv\n"
                "========================================"
            )


        except Exception as e:

            print(
                "\n"
                "========================================\n"
                "GAGAL MENYIMPAN EVALUASI REALTIME\n"
                "========================================\n"
                f"Error: {e}\n"
                "========================================"
            )


    # Reset scan
    def reset_scan():

        st.session_state.live_result = None

        st.session_state.live_error = None

        st.session_state.live_barcode = None


        with shared.lock:

            # Reset single trigger
            shared.locked = False

            shared.barcode_detected = None

            shared.barcode_frame = None

            shared.barcode_frame_annotated = None

            shared.last_capture_time = 0


            # Reset evaluasi realtime
            shared.frame_count = 0

            shared.scan_start_time = None

            shared.scan_start_datetime = None

            shared.detection_frame = None

            shared.detection_time = None

            shared.processed_frame = None

            shared.processed_time = None

            shared.total_processing_time = None

            shared.fps = 0.0

            shared.fps_frame_count = 0

            shared.fps_start_time = (
                time.perf_counter()
            )

            shared.frame_records = []

            shared.scan_id = None

            shared.evaluation_saved = False


        print(
            "\n"
            "========================================\n"
            "SCAN DI-RESET\n"
            "Menunggu scan produk berikutnya...\n"
            "========================================"
        )


    # Kolom hasil
    with col_result:
        # Reset
        st.button(
            "Scan Produk Lain",
            icon=":material/refresh:",
            on_click=reset_scan,
            width="stretch"
        )

        st.subheader("Hasil Proses")


        if auto_mode:

            # Informasi realtime
            realtime_placeholder = st.empty()


            with shared.lock:

                current_frame_display = (
                    shared.frame_count
                )

                current_fps_display = (
                    shared.fps
                )


            if current_fps_display > 0:

                realtime_placeholder.info(
                    f":material/bar_chart: FPS aktual: "
                    f"{current_fps_display:.2f} FPS\n\n"
                    f":material/movie: Frame saat ini: "
                    f"{current_frame_display}"
                )

            else:

                realtime_placeholder.info(
                    f":material/bar_chart: FPS aktual: menghitung...\n\n"
                    f":material/movie: Frame saat ini: "
                    f"{current_frame_display}"
                )


            # Caption barcode
            barcode_caption = st.empty()


            if (
                st.session_state.live_barcode
                is not None
            ):

                barcode_caption.caption(
                    ":material/photo_camera: Barcode terdeteksi otomatis: "
                    f"`{st.session_state.live_barcode}`"
                )


            # Container hasil
            placeholder = st.container()


            # Langkah 1: proses barcode yang terdeteksi
            if (
                st.session_state.live_result is None
                and
                st.session_state.live_error is None
            ):

                with shared.lock:

                    pending_barcode = (
                        shared.barcode_detected
                    )

                    pending_frame = (
                        shared.barcode_frame
                    )

                    pending_frame_annotated = (
                        shared.barcode_frame_annotated
                    )

                    pending_detection_frame = (
                        shared.detection_frame
                    )

                    pending_detection_time = (
                        shared.detection_time
                    )


                if (
                    pending_barcode is not None
                    and
                    pending_frame is not None
                ):

                    # Lock
                    with shared.lock:

                        shared.locked = True


                    # Frame diproses = frame yang sama saat barcode ditemukan
                    with shared.lock:

                        shared.processed_frame = (
                            pending_detection_frame
                        )

                        shared.processed_time = (
                            time.perf_counter()
                            - shared.scan_start_time
                        )


                    # Simpan frame BERSIH -- ini yang diproses
                    temp_path = "temp_live.jpg"

                    cv2.imwrite(
                        temp_path,
                        pending_frame
                    )

                    # Simpan frame BERANOTASI (kalau ada) -- untuk arsip
                    archive_path = None

                    if pending_frame_annotated is not None:

                        archive_path = "temp_live_annotated.jpg"

                        cv2.imwrite(
                            archive_path,
                            pending_frame_annotated
                        )


                    # Terminal
                    print(
                        "\n"
                        "========================================\n"
                        "FRAME DIPROSES\n"
                        "========================================\n"
                        f"Barcode           : "
                        f"{pending_barcode}\n"
                        f"Detection frame   : "
                        f"{pending_detection_frame}\n"
                        f"Detection time    : "
                        f"{pending_detection_time:.3f} detik\n"
                        f"Processed frame   : "
                        f"{pending_detection_frame}\n"
                        f"Processed time    : "
                        f"{shared.processed_time:.3f} detik\n"
                        f"FPS aktual        : "
                        f"{shared.fps:.2f} FPS\n"
                        "========================================"
                    )


                    # Proses sistem
                    try:

                        processing_start = (
                            time.perf_counter()
                        )


                        with placeholder:

                            with st.spinner(
                                "Memproses Vision AI..."
                            ):

                                result = (
                                    product_service.process_image(
                                        temp_path,
                                        mode="Live Camera Otomatis",
                                        archive_image_path=archive_path,
                                    )
                                )


                        processing_end = (
                            time.perf_counter()
                        )


                        # Total waktu processing
                        total_processing_time = (
                            processing_end
                            - processing_start
                        )


                        with shared.lock:

                            shared.total_processing_time = (
                                total_processing_time
                            )


                        st.session_state.live_result = (
                            result
                        )


                        # Barcode final
                        vision_info = result.get(
                            "vision",
                            {}
                        )


                        barcode_final = (
                            vision_info.get(
                                "barcode_pyzbar"
                            )
                            or
                            vision_info.get(
                                "barcode_vision_asli"
                            )
                        )


                        st.session_state.live_barcode = (
                            barcode_final
                            or
                            pending_barcode
                        )


                        # Simpan evaluasi
                        save_realtime_evaluation()


                    except Exception as e:

                        # Simpan waktu processing jika error
                        processing_end = (
                            time.perf_counter()
                        )


                        with shared.lock:

                            shared.total_processing_time = (
                                processing_end
                                - processing_start
                            )


                        st.session_state.live_error = (
                            str(e)
                        )


                else:

                    with placeholder:

                        st.info(
                            "Menunggu barcode terdeteksi "
                            "di kamera..."
                        )


            # Langkah 2: tampilkan hasil
            if (
                st.session_state.live_result
                is not None
            ):

                # Hasil sistem
                show_result(
                    st.session_state.live_result,
                    placeholder
                )


                # Ambil data evaluasi terakhir
                with shared.lock:

                    final_detection_frame = (
                        shared.detection_frame
                    )

                    final_detection_time = (
                        shared.detection_time
                    )

                    final_processed_frame = (
                        shared.processed_frame
                    )

                    final_processed_time = (
                        shared.processed_time
                    )

                    final_fps = (
                        shared.fps
                    )

                    final_processing_time = (
                        shared.total_processing_time
                    )

                    final_frames_checked = (
                        len(shared.frame_records)
                    )

                    final_scan_id = (
                        shared.scan_id
                    )
                    # Ambil kondisi pengujian otomatis dari hasil vision
                    vision_info_final = (
                        st.session_state.live_result.get("vision", {})
                        if st.session_state.live_result
                        else {}
                    )

                    final_kondisi_pencahayaan = vision_info_final.get(
                        "kondisi_pencahayaan_auto", "-"
                    )

                    final_kondisi_fisik = vision_info_final.get(
                        "kondisi_fisik_barcode_auto", "-"
                    )

                    final_detail_level = vision_info_final.get(
                        "image_detail_level_auto", "-"
                    )


                # Data evaluasi realtime
                st.divider()

                st.subheader(
                    ":material/bar_chart: Data Evaluasi Realtime"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Scan ID",
                        str(final_scan_id)
                        if final_scan_id is not None
                        else "-"
                    )

                    st.metric(
                        "FPS Aktual",
                        f"{final_fps:.2f}"
                    )

                    if final_detection_frame is not None:
                        st.metric(
                            "Frame Deteksi",
                            str(final_detection_frame)
                        )

                    if final_processed_frame is not None:
                        st.metric(
                            "Frame Diproses",
                            str(final_processed_frame)
                        )


                with col2:

                    st.metric(
                        "Frame Diperiksa",
                        str(final_frames_checked)
                    )

                    if final_detection_time is not None:
                        st.metric(
                            "Waktu Deteksi",
                            f"{final_detection_time:.3f} s"
                        )

                    if final_processed_time is not None:
                        st.metric(
                            "Waktu Frame Diproses",
                            f"{final_processed_time:.3f} s"
                        )

                    if final_processing_time is not None:
                        st.metric(
                            "Total Processing",
                            f"{final_processing_time:.3f} s"
                        )

                # --- KELUAR dari col1/col2, sejajar dengan "col1, col2 = st.columns(2)" ---

                # Kondisi pengujian otomatis
                st.divider()

                st.subheader(
                    ":material/tune: Kondisi Pengujian (Auto-Deteksi)"
                )

                col3, col4, col5 = st.columns(3)

                with col3:

                    icon_pencahayaan = (
                        ":material/light_mode:"
                        if final_kondisi_pencahayaan == "terang"
                        else ":material/dark_mode:"
                    )

                    st.metric(
                        "Pencahayaan",
                        final_kondisi_pencahayaan.capitalize()
                        if final_kondisi_pencahayaan != "-"
                        else "-"
                    )

                    if final_kondisi_pencahayaan == "redup":
                        st.caption(f"{icon_pencahayaan} Cahaya kurang -- pertimbangkan pencahayaan tambahan")
                    elif final_kondisi_pencahayaan == "terang":
                        st.caption(f"{icon_pencahayaan} Cahaya cukup")


                with col4:

                    st.metric(
                        "Kualitas Fisik Barcode",
                        final_kondisi_fisik.replace("_", " ").capitalize()
                        if final_kondisi_fisik != "-"
                        else "-"
                    )

                    if final_kondisi_fisik == "rusak_pudar":
                        st.caption(":material/warning: Kontras/ketajaman rendah -- kualitas barcode kurang baik")
                    elif final_kondisi_fisik == "mulus":
                        st.caption(":material/check_circle: Barcode dalam kondisi baik")


                with col5:

                    st.metric(
                        "Image Detail Level",
                        final_detail_level.upper()
                        if final_detail_level != "-"
                        else "-"
                    )

                    if final_detail_level == "high":
                        st.caption(":material/hd: Detail tinggi dikirim ke Vision AI (kompensasi kondisi kurang ideal)")
                    elif final_detail_level == "low":
                        st.caption(":material/sd: Detail standar (kondisi gambar sudah baik)")

                # Status penyimpanan
                with shared.lock:

                    evaluation_saved = (
                        shared.evaluation_saved
                    )


                if evaluation_saved:

                    st.success(
                        ":material/check_circle: Data evaluasi realtime "
                        "telah disimpan ke CSV."
                    )


            # Error
            elif (
                st.session_state.live_error
                is not None
            ):

                with placeholder:

                    st.error(
                        f"Gagal memproses: "
                        f"{st.session_state.live_error}\n\n"
                        "Kemungkinan penyebab: terlalu banyak "
                        "permintaan ke API dalam waktu singkat "
                        "(rate limit) atau koneksi bermasalah."
                    )


                st.button(
                    "Coba Scan Lagi",
                    icon=":material/refresh:",
                    on_click=reset_scan,
                    width="stretch"
                )


        # Mode manual
        else:

            if capture_clicked:

                # Ambil data frame
                with shared.lock:

                    pending_barcode = (
                        shared.barcode_detected
                    )

                    pending_frame = (
                        shared.barcode_frame
                    )

                    pending_frame_annotated = (
                        shared.barcode_frame_annotated
                    )

                    fallback_frame = (
                        shared.latest_frame
                    )


                if (
                    pending_barcode is not None
                    and
                    pending_frame is not None
                ):

                    st.caption(
                        f"Barcode terdeteksi: "
                        f"`{pending_barcode}`"
                    )

                    frame_to_process = (
                        pending_frame
                    )

                    archive_frame = (
                        pending_frame_annotated
                    )


                elif fallback_frame is not None:

                    st.caption(
                        "Belum ada barcode yang terdeteksi "
                        "stabil oleh pyzbar — tetap diproses "
                        "via Vision AI dari frame kamera terakhir."
                    )

                    frame_to_process = (
                        fallback_frame
                    )

                    # Frame fallback tidak pernah digambar kotak,
                    # jadi tidak ada versi beranotasi untuk ini.
                    archive_frame = None


                else:

                    frame_to_process = None

                    archive_frame = None


                # Proses frame
                if frame_to_process is not None:

                    try:

                        process_frame_and_show(
                            frame_to_process,
                            st.container(),
                            archive_frame_bgr=archive_frame,
                        )


                    except Exception as e:

                        st.error(
                            f"Gagal memproses: {e}\n\n"
                            "Kemungkinan penyebab: rate limit API "
                            "atau koneksi bermasalah. Tunggu beberapa "
                            "detik lalu coba capture ulang."
                        )


                    finally:

                        with shared.lock:

                            shared.barcode_detected = None

                            shared.barcode_frame = None

                            shared.barcode_frame_annotated = None


                else:

                    st.warning(
                        "Belum ada frame kamera yang tertangkap. "
                        "Pastikan kamera sudah aktif."
                    )


    # Auto refresh mode otomatis
    if (
        auto_mode
        and
        st.session_state.live_result is None
        and
        st.session_state.live_error is None
    ):

        time.sleep(1)

        st.rerun()