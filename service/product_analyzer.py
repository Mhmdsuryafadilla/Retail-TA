from decimal import Decimal


class ProductAnalyzer:

    def safe_number(self, value):
        try:

            if value is None:
                return 0.0

            if isinstance(value, Decimal):
                return float(value)

            return float(value)

        except:
            return 0.0

    def detect_unit(self, quantity: str):

        if not quantity:
            return "unknown"

        qty = quantity.lower().replace(" ", "")

        if "kg" in qty:
            return "kg"

        if "g" in qty:
            return "g"

        if "ml" in qty:
            return "ml"

        if "l" in qty:
            return "l"

        return "unknown"

    def calculate_unit_price(
        self,
        price,
        qty,
        unit
    ):

        if qty <= 0:
            return None

        if unit == "kg":

            return round(price / qty, 2), "Rp/kg"

        elif unit == "g":

            return round(price / qty, 2), "Rp/g"

        elif unit == "l":

            return round(price / qty, 2), "Rp/L"

        elif unit == "ml":

            return round(price / qty, 2), "Rp/ml"

        return None

    def analyze(
        self,
        product,
        similar_products
    ):

        result = {}

        harga = self.safe_number(
            product.get("harga")
        )

        qty = self.safe_number(
            product.get("qty_val")
        )

        quantity = product.get(
            "kuantitas",
            ""
        )

        unit = self.detect_unit(quantity)

        # Hitung harga per satuan (kg/g/l/ml)
        unit_price = self.calculate_unit_price(
            harga,
            qty,
            unit
        )

        if unit_price:

            result["harga_per_satuan"] = unit_price[0]

            result["satuan"] = unit_price[1]

        else:

            result["harga_per_satuan"] = None

            result["satuan"] = "-"

        # Cari produk termurah & termahal dari produk serupa,
        # sekaligus hitung berapa yang lebih murah/mahal dari produk ini
        cheapest = None
        cheapest_price = None

        most_expensive = None
        expensive_price = None

        lower = 0
        higher = 0

        prices = []

        for item in similar_products:

            item_price = self.safe_number(
                item.get("harga")
            )

            prices.append(item_price)

            if cheapest_price is None or item_price < cheapest_price:

                cheapest_price = item_price

                cheapest = item

            if expensive_price is None or item_price > expensive_price:

                expensive_price = item_price

                most_expensive = item

            if item_price < harga:

                lower += 1

            elif item_price > harga:

                higher += 1

        result["produk_termurah"] = cheapest

        result["produk_termahal"] = most_expensive

        result["jumlah_produk_serupa"] = len(similar_products)

        result["lebih_murah_dari"] = lower

        result["lebih_mahal_dari"] = higher

        # Ranking harga produk ini di antara semua produk serupa
        all_prices = prices.copy()

        all_prices.append(harga)

        all_prices.sort()

        try:

            ranking = all_prices.index(harga) + 1

        except:

            ranking = "-"

        result["ranking_harga"] = ranking

        # Harga rata-rata dari semua produk serupa + produk ini
        if len(all_prices) > 0:

            avg = sum(all_prices) / len(all_prices)

        else:

            avg = 0

        result["harga_rata_rata"] = round(avg, 2)

        # Selisih harga dengan produk termurah
        if cheapest_price is not None:

            result["selisih_termurah"] = round(

                harga - cheapest_price,

                2

            )

        else:

            result["selisih_termurah"] = 0

        # Kategori harga: Murah / Sedang / Mahal berdasarkan rata-rata
        if avg == 0:

            kategori = "Tidak diketahui"

        elif harga <= avg * 0.90:

            kategori = "Murah"

        elif harga >= avg * 1.10:

            kategori = "Mahal"

        else:

            kategori = "Sedang"

        result["kategori_harga"] = kategori

        # Penilaian value for money berdasarkan kategori harga
        if result["harga_per_satuan"] is None:

            value = "Tidak dapat dianalisis"

        elif kategori == "Murah":

            value = "Sangat Baik"

        elif kategori == "Sedang":

            value = "Baik"

        else:

            value = "Cukup"

        result["value_for_money"] = value

        # Ambil 5 produk alternatif teratas
        result["alternatif"] = similar_products[:5]

        return result