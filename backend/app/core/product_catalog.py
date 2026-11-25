"""Persistent store-owned product master for checkout decisions."""

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

from backend.app.config import settings
from backend.app.core.live_pricing import STORE_CATALOG
from backend.app.schemas import NutritionData


class ProductCatalog:
    def __init__(self, database_path: str):
        self.database_path = database_path
        directory = os.path.dirname(database_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._create_schema()
        self._seed_if_empty()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _create_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    sku TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    price REAL NOT NULL CHECK(price >= 0),
                    currency TEXT NOT NULL,
                    tax_code TEXT NOT NULL,
                    barcode TEXT,
                    nutrition_json TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _seed_if_empty(self) -> None:
        with self._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            if count:
                return
            for sku, product in STORE_CATALOG.items():
                nutrition = product.get("nutrition")
                nutrition_data = nutrition.model_dump() if nutrition else None
                connection.execute(
                    """
                    INSERT INTO products (sku, name, category, price, currency, tax_code, barcode, nutrition_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sku,
                        product["name"],
                        product["category"],
                        product["default_price"],
                        settings.CURRENCY_CODE,
                        "GST_5",
                        product.get("barcode"),
                        json.dumps(nutrition_data),
                    ),
                )

    def list_products(self) -> Iterable[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT sku, name, category, price, currency, tax_code, barcode, nutrition_json, active FROM products ORDER BY name"
            ).fetchall()
        return [self._row_to_product(row) for row in rows]

    def get_product(self, sku: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT sku, name, category, price, currency, tax_code, barcode, nutrition_json, active FROM products WHERE sku = ?",
                (sku,),
            ).fetchone()
        return self._row_to_product(row) if row else None

    def upsert_product(self, sku: str, name: str, category: str, price: float) -> Dict[str, Any]:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO products (sku, name, category, price, currency, tax_code)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sku) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    price = excluded.price,
                    active = 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (sku, name, category, price, settings.CURRENCY_CODE, "GST_5"),
            )
        product = self.get_product(sku)
        if product is None:
            raise RuntimeError("Product catalogue write was not persisted")
        return product

    @staticmethod
    def _row_to_product(row: sqlite3.Row) -> Dict[str, Any]:
        nutrition = json.loads(row["nutrition_json"]) if row["nutrition_json"] else None
        return {
            "sku": row["sku"],
            "name": row["name"],
            "category": row["category"],
            "price": row["price"],
            "currency": row["currency"],
            "tax_code": row["tax_code"],
            "barcode": row["barcode"],
            "nutrition": NutritionData(**nutrition).model_dump() if nutrition else None,
            "active": bool(row["active"]),
        }


product_catalog = ProductCatalog(settings.CATALOG_DATABASE_PATH)
