"""Seed catalog and inventory.

In a real deployment these would be a product service / PostgreSQL and an
inventory API. They are kept in-process so the project runs with no external
dependencies; the tool layer is the only code that touches them.
"""
from __future__ import annotations

from pydantic import BaseModel


class Product(BaseModel):
    id: str
    name: str
    brand: str
    category: str
    price: float          # INR
    rating: float         # 0-5
    tags: list[str]
    description: str


PRODUCTS: list[Product] = [
    Product(id="AUD-001", name="Sony WH-CH720N Wireless Headphones", brand="Sony", category="audio",
            price=8990, rating=4.4, tags=["wireless", "bluetooth", "noise cancelling", "over-ear"],
            description="Lightweight over-ear headphones with active noise cancelling and 35h battery."),
    Product(id="AUD-002", name="boAt Rockerz 450 Wireless Headphones", brand="boAt", category="audio",
            price=1499, rating=4.0, tags=["wireless", "bluetooth", "on-ear", "budget"],
            description="Budget on-ear bluetooth headphones with 15h playback."),
    Product(id="AUD-003", name="JBL Tune 770NC Wireless Headphones", brand="JBL", category="audio",
            price=6999, rating=4.3, tags=["wireless", "bluetooth", "noise cancelling", "over-ear"],
            description="Adaptive noise cancelling headphones with 70h battery life."),
    Product(id="AUD-004", name="OnePlus Buds 3 Wireless Earbuds", brand="OnePlus", category="audio",
            price=5499, rating=4.2, tags=["wireless", "earbuds", "noise cancelling", "tws"],
            description="True wireless earbuds with 49dB noise cancellation."),
    Product(id="AUD-005", name="Sennheiser HD 450BT Wireless Headphones", brand="Sennheiser", category="audio",
            price=9990, rating=4.5, tags=["wireless", "bluetooth", "noise cancelling", "over-ear"],
            description="Premium sound with noise cancelling and 30h battery."),
    Product(id="LAP-001", name="Lenovo IdeaPad Slim 3 Laptop", brand="Lenovo", category="laptops",
            price=52990, rating=4.2, tags=["laptop", "student", "ryzen 5", "16gb"],
            description="15.6 inch laptop with Ryzen 5, 16GB RAM and 512GB SSD."),
    Product(id="LAP-002", name="ASUS Vivobook 15 Laptop", brand="ASUS", category="laptops",
            price=45990, rating=4.0, tags=["laptop", "student", "core i5", "8gb"],
            description="Everyday laptop with Intel Core i5 and 8GB RAM."),
    Product(id="LAP-003", name="Apple MacBook Air M2 Laptop", brand="Apple", category="laptops",
            price=89900, rating=4.7, tags=["laptop", "macbook", "m2", "developer"],
            description="13.6 inch MacBook Air with Apple M2 chip and all-day battery."),
    Product(id="LAP-004", name="HP Victus Gaming Laptop", brand="HP", category="laptops",
            price=67990, rating=4.3, tags=["laptop", "gaming", "rtx 3050", "16gb"],
            description="Gaming laptop with RTX 3050 GPU and 144Hz display."),
    Product(id="PHN-001", name="Samsung Galaxy A55 Phone", brand="Samsung", category="phones",
            price=39999, rating=4.3, tags=["phone", "android", "5g", "amoled"],
            description="5G Android phone with AMOLED display and 50MP camera."),
    Product(id="PHN-002", name="Google Pixel 8a Phone", brand="Google", category="phones",
            price=52999, rating=4.5, tags=["phone", "android", "5g", "camera"],
            description="Pixel phone with flagship camera and 7 years of updates."),
    Product(id="PHN-003", name="Redmi Note 13 Phone", brand="Xiaomi", category="phones",
            price=17999, rating=4.1, tags=["phone", "android", "5g", "budget"],
            description="Budget 5G phone with 108MP camera."),
    Product(id="SHO-001", name="Nike Revolution 7 Running Shoes", brand="Nike", category="footwear",
            price=3695, rating=4.2, tags=["shoes", "running", "sports"],
            description="Cushioned running shoes for daily training."),
    Product(id="SHO-002", name="ASICS Gel-Contend 9 Running Shoes", brand="ASICS", category="footwear",
            price=4499, rating=4.4, tags=["shoes", "running", "gel", "sports"],
            description="Gel cushioning running shoes for neutral runners."),
    Product(id="KIT-001", name="Philips Drip Coffee Maker", brand="Philips", category="kitchen",
            price=2795, rating=4.1, tags=["coffee", "drip", "machine"],
            description="Drip coffee maker with 1.2L jug."),
    Product(id="KIT-002", name="De'Longhi Espresso Coffee Machine", brand="De'Longhi", category="kitchen",
            price=14990, rating=4.4, tags=["coffee", "espresso", "machine"],
            description="Pump espresso machine with milk frother."),
]

PRODUCTS_BY_ID: dict[str, Product] = {p.id: p for p in PRODUCTS}

# product_id -> (quantity, warehouse)
INVENTORY: dict[str, tuple[int, str]] = {
    "AUD-001": (12, "BLR-1"),
    "AUD-002": (40, "BLR-1"),
    "AUD-003": (0, "DEL-2"),     # out of stock: exercises the "skip unavailable" path
    "AUD-004": (7, "BLR-1"),
    "AUD-005": (3, "MUM-1"),
    "LAP-001": (5, "BLR-1"),
    "LAP-002": (0, "DEL-2"),
    "LAP-003": (2, "MUM-1"),
    "LAP-004": (4, "BLR-1"),
    "PHN-001": (9, "BLR-1"),
    "PHN-002": (6, "MUM-1"),
    "PHN-003": (25, "DEL-2"),
    "SHO-001": (0, "BLR-1"),
    "SHO-002": (0, "MUM-1"),     # both shoes out of stock: exercises "none available"
    "KIT-001": (14, "BLR-1"),
    "KIT-002": (3, "DEL-2"),
}
