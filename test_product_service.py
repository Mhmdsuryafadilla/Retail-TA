from service.product_service import ProductService
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(
    BASE_DIR,
    "uploads",
    "test.jpg"
)

print("=" * 70)
print("TEST PRODUCT SERVICE")
print("=" * 70)
print("[INFO] Gambar :", image_path)

service = ProductService()

hasil = service.process_image(
    image_path=image_path,
    mode="Testing"
)

print("\n" + "=" * 70)
print("HASIL AKHIR")
print("=" * 70)

print("Vision :", hasil["vision"])
print("Product:", hasil["product"])
print("Answer :", hasil["answer"])

print("\nTEST SELESAI")