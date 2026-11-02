# test_pyzbar_env.py
import pyzbar
print("pyzbar version info:", pyzbar.__file__)

from pyzbar.pyzbar import decode
import cv2

img = cv2.imread("temp2.jpeg")  # pakai file temp.jpg yg baru di-generate app-mu
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = decode(gray)
print("Hasil decode:", result)