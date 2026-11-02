from database.product_repository import ProductRepository

repo = ProductRepository()

produk = repo.get_all_products()

print(len(produk))

print(produk[0])

repo.close()