

def validate_ean13(barcode: str) -> bool:
    """Validasi checksum EAN-13 (13 digit)."""
    if not barcode.isdigit() or len(barcode) != 13:
        return False

    digits = [int(d) for d in barcode]
    checksum = 0
    for i, d in enumerate(digits[:12]):
        weight = 1 if i % 2 == 0 else 3
        checksum += d * weight

    check_digit = (10 - (checksum % 10)) % 10
    return check_digit == digits[12]


def validate_upca(barcode: str) -> bool:
    """Validasi checksum UPC-A (12 digit)."""
    if not barcode.isdigit() or len(barcode) != 12:
        return False

    digits = [int(d) for d in barcode]
    checksum = 0
    for i, d in enumerate(digits[:11]):
        weight = 3 if i % 2 == 0 else 1
        checksum += d * weight

    check_digit = (10 - (checksum % 10)) % 10
    return check_digit == digits[11]


def validate_checksum(barcode: str) -> bool:

    if not barcode:
        return False

    barcode = barcode.strip()

    if len(barcode) == 13:
        return validate_ean13(barcode)
    elif len(barcode) == 12:
        return validate_upca(barcode)
    else:
        return False