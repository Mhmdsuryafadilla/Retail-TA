import re


class BarcodeValidator:
    """
    Validator untuk barcode EAN-13.

    Fitur:
    - Menghapus spasi
    - Menghapus karakter selain angka
    - Validasi panjang
    - Validasi checksum EAN-13
    """

    @staticmethod
    def clean(barcode: str) -> str:
        """
        Menghapus semua karakter selain angka.
        """

        if barcode is None:
            return ""

        return re.sub(r"\D", "", str(barcode))


    @staticmethod
    def is_numeric(barcode: str) -> bool:
        """
        Memastikan barcode hanya berisi angka.
        """

        return barcode.isdigit()


    @staticmethod
    def is_ean13(barcode: str) -> bool:
        """
        Memastikan panjang barcode 13 digit.
        """

        return len(barcode) == 13


    @staticmethod
    def calculate_checksum(barcode12: str) -> int:
        """
        Menghitung digit checksum EAN-13
        menggunakan 12 digit pertama.
        """

        total = 0

        for i, digit in enumerate(barcode12):

            digit = int(digit)

            if i % 2 == 0:
                total += digit
            else:
                total += digit * 3

        checksum = (10 - (total % 10)) % 10

        return checksum


    @classmethod
    def validate(cls, barcode: str):
        """
        Return:

        {
            "valid": True/False,
            "barcode": "...",
            "message": "..."
        }
        """

        barcode = cls.clean(barcode)

        if barcode == "":
            return {
                "valid": False,
                "barcode": "",
                "message": "Barcode kosong."
            }

        if not cls.is_numeric(barcode):
            return {
                "valid": False,
                "barcode": barcode,
                "message": "Barcode harus berupa angka."
            }

        if not cls.is_ean13(barcode):
            return {
                "valid": False,
                "barcode": barcode,
                "message": "Barcode bukan format EAN-13."
            }

        checksum = cls.calculate_checksum(barcode[:-1])

        if checksum != int(barcode[-1]):
            return {
                "valid": False,
                "barcode": barcode,
                "message": "Checksum barcode tidak valid."
            }

        return {
            "valid": True,
            "barcode": barcode,
            "message": "Barcode valid."
        }