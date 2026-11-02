from database.connection import get_connection


class ProductRepository:
 
    SELECT_COLUMNS = """
        barcode,
        nama_produk,
        brand,
        kategori_utama,
        sub_kategori,
        jenis_spesifik,
        kuantitas,
        qty_val,
        satuan,
        negara,
        harga,
        harga AS harga_riset
    """

    def __init__(self):
        self.conn = get_connection()
        self.cursor = self.conn.cursor(dictionary=True)

    def get_all_products(self):
        sql = f"""
        SELECT {self.SELECT_COLUMNS}
        FROM produk
        """
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def get_by_name(self, nama_produk):
        sql = f"""
        SELECT {self.SELECT_COLUMNS}
        FROM produk
        WHERE nama_produk LIKE %s
        LIMIT 1
        """
        self.cursor.execute(sql, (f"%{nama_produk}%",))
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_by_barcode(self, barcode):
        sql = f"""
        SELECT {self.SELECT_COLUMNS}
        FROM produk
        WHERE barcode=%s
        LIMIT 1
        """
        self.cursor.execute(sql, (barcode,))
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_similar_products(self, product):
        sub_kategori = product.get("sub_kategori")
        jenis_spesifik = product.get("jenis_spesifik")
        satuan = product.get("satuan")
        barcode = product.get("barcode")

        try:
            harga = float(product.get("harga") or 0)
        except (TypeError, ValueError):
            harga = 0

        sql = f"""
        SELECT {self.SELECT_COLUMNS}
        FROM produk
        WHERE
            sub_kategori = %s
            AND barcode != %s
            AND CAST(harga AS DECIMAL(15,2)) < %s
        """
        params = [sub_kategori, barcode, harga]

        if jenis_spesifik:
            sql += " AND jenis_spesifik = %s"
            params.append(jenis_spesifik)

        if satuan:
            sql += " AND satuan = %s"
            params.append(satuan)

        sql += " ORDER BY CAST(harga AS DECIMAL(15,2)) ASC LIMIT 10"

        self.cursor.execute(sql, tuple(params))
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    # Menghitung TOTAL produk pembanding yang SEBENARNYA valid di database

    def get_similar_products_count(self, product):
        sub_kategori = product.get("sub_kategori")
        jenis_spesifik = product.get("jenis_spesifik")
        satuan = product.get("satuan")
        barcode = product.get("barcode")

        try:
            harga = float(product.get("harga") or 0)
        except (TypeError, ValueError):
            harga = 0

        sql = """
        SELECT COUNT(*) AS total
        FROM produk
        WHERE
            sub_kategori = %s
            AND barcode != %s
            AND CAST(harga AS DECIMAL(15,2)) < %s
        """
        params = [sub_kategori, barcode, harga]

        if jenis_spesifik:
            sql += " AND jenis_spesifik = %s"
            params.append(jenis_spesifik)

        if satuan:
            sql += " AND satuan = %s"
            params.append(satuan)

        self.cursor.execute(sql, tuple(params))
        row = self.cursor.fetchone()
        return row["total"] if row else 0

    def close(self):
        self.cursor.close()
        self.conn.close()