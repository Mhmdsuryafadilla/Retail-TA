from vision.vision_api import VisionAPI
import os

# Path gambar test
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "uploads", "test.jpg")

vision = VisionAPI()

if not os.path.exists(image_path):
    print(f"[ERROR] File gambar tidak ditemukan:")
    print(image_path)
    raise SystemExit(1)

print("TEST VISION AI")
print(f"[INFO] Gambar : {image_path}")
print("[INFO] Memanggil Vision AI...")
print()

try:
    hasil = vision.identify_product(image_path)
except Exception as e:
    print()
    print("[ERROR] Vision AI gagal")
    print(e)
    raise SystemExit(1)

print()
print("HASIL IDENTIFIKASI")
print(hasil)

print()
print("VISION AI USAGE / TOKEN")

if vision.last_usage:
    print("Usage lengkap:")
    print(vision.last_usage)
    print()

    total_tokens = vision.last_usage.get("total_tokens")
    prompt_tokens = vision.last_usage.get("prompt_tokens")
    completion_tokens = vision.last_usage.get("completion_tokens")

    print(f"Prompt tokens     : {prompt_tokens}")
    print(f"Completion tokens : {completion_tokens}")
    print(f"Total tokens      : {total_tokens}")
else:
    print("[WARNING] Vision AI tidak mengembalikan informasi usage/token.")

print()
print("TEST SELESAI")