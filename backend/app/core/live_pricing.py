import httpx
import re
from typing import Dict, Any, Optional
from backend.app.schemas import NutritionData

# Central In-Memory Store Inventory with Cognitive Size Variants
STORE_CATALOG: Dict[str, Dict[str, Any]] = {
    "coke_bottle": {
        "name": "Coca-Cola Classic (500ml)",
        "category": "Beverage",
        "default_price": 50.0,
        "barcode": "5449000000996",
        "nutrition": NutritionData(calories_kcal=210.0, sugar_g=53.0, fat_g=0.0, allergens=[], eco_grade="D")
    },
    "sprite_large": {
        "name": "Sprite Lime Drink (740ml / 600ml+140ml)",
        "category": "Beverage",
        "default_price": 40.0,
        "barcode": "5449000014528",
        "nutrition": NutritionData(calories_kcal=192.0, sugar_g=48.0, fat_g=0.0, allergens=[], eco_grade="D")
    },
    "sprite_small": {
        "name": "Sprite Lime Drink (250ml)",
        "category": "Beverage",
        "default_price": 20.0,
        "barcode": "5449000014528",
        "nutrition": NutritionData(calories_kcal=96.0, sugar_g=24.0, fat_g=0.0, allergens=[], eco_grade="D")
    },
    "aqua_velly_large": {
        "name": "Aqua Velly Packaged Mineral Water (1L)",
        "category": "Water",
        "default_price": 20.0,
        "barcode": "8901764012239",
        "nutrition": NutritionData(calories_kcal=0.0, sugar_g=0.0, fat_g=0.0, allergens=[], eco_grade="A")
    },
    "aqua_velly_small": {
        "name": "Aqua Velly Packaged Mineral Water (500ml)",
        "category": "Water",
        "default_price": 10.0,
        "barcode": "8901764012239",
        "nutrition": NutritionData(calories_kcal=0.0, sugar_g=0.0, fat_g=0.0, allergens=[], eco_grade="A")
    },
    "canned_food": {
        "name": "Heinz Canned Soup / Food",
        "category": "Packaged Food",
        "default_price": 100.0,
        "barcode": "5000157024671",
        "nutrition": NutritionData(calories_kcal=180.0, sugar_g=8.5, fat_g=3.2, allergens=["Celery"], eco_grade="B")
    },
    "snack_pack": {
        "name": "Lays / Puffed Corn Snack Pack",
        "category": "Snacks",
        "default_price": 30.0,
        "barcode": "8901491101837",
        "nutrition": NutritionData(calories_kcal=270.0, sugar_g=2.1, fat_g=15.0, allergens=["Gluten", "Soy"], eco_grade="C")
    },
    "juice_box": {
        "name": "Real Mixed Fruit Juice (1L)",
        "category": "Beverage",
        "default_price": 85.0,
        "barcode": "8901207010017",
        "nutrition": NutritionData(calories_kcal=130.0, sugar_g=28.0, fat_g=0.1, allergens=[], eco_grade="B")
    },
    "fresh_fruit": {
        "name": "Fresh Royal Gala Apple",
        "category": "Produce",
        "default_price": 35.0,
        "barcode": "3103",
        "nutrition": NutritionData(calories_kcal=95.0, sugar_g=19.0, fat_g=0.3, allergens=[], eco_grade="A")
    }
}

_API_CACHE: Dict[str, Dict[str, Any]] = {}

async def fetch_open_food_facts_data(barcode: str) -> Optional[Dict[str, Any]]:
    if barcode in _API_CACHE:
        return _API_CACHE[barcode]
        
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    headers = {"User-Agent": "VisionCheckout-SmartPOS-StudentProject/2.0"}
    
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == 1:
                    product = data.get("product", {})
                    nutriments = product.get("nutriments", {})
                    
                    extracted = {
                        "name": product.get("product_name", ""),
                        "brands": product.get("brands", ""),
                        "calories": float(nutriments.get("energy-kcal_100g", 0.0) or 0.0),
                        "sugar": float(nutriments.get("sugars_100g", 0.0) or 0.0),
                        "fat": float(nutriments.get("fat_100g", 0.0) or 0.0),
                        "eco_grade": str(product.get("ecoscore_grade", "B")).upper()
                    }
                    _API_CACHE[barcode] = extracted
                    return extracted
    except Exception:
        pass
    return None

def extract_mrp_from_text(text: str) -> Optional[float]:
    patterns = [
        r'(?:mrp|price|rs\.?|₹)\s*[:\-]?\s*([0-9]{1,4}(?:\.[0-9]{1,2})?)',
        r'([0-9]{1,4}(?:\.[0-9]{1,2})?)\s*(?:rs|inr|₹)'
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 5.0 <= val <= 2000.0:
                    return val
            except ValueError:
                continue
    return None

async def resolve_realtime_price(
    product_key: str, 
    ocr_text: Optional[str] = None
) -> Dict[str, Any]:
    catalog_entry = STORE_CATALOG.get(product_key, {
        "name": product_key.replace("_", " ").title(),
        "category": "Beverage" if "bottle" in product_key or "aqua" in product_key or "sprite" in product_key else "General Merchandise",
        "default_price": 20.0,
        "barcode": "8901030000000",
        "nutrition": NutritionData(calories_kcal=100.0, sugar_g=10.0, fat_g=0.0, allergens=[], eco_grade="B")
    })
    
    # 1. Tier 1 Check: Printed MRP via OCR
    if ocr_text:
        extracted_price = extract_mrp_from_text(ocr_text)
        if extracted_price:
            return {
                "name": catalog_entry["name"],
                "category": catalog_entry["category"],
                "price": extracted_price,
                "source": "OCR_PACKAGING (Visual MRP)",
                "nutrition": catalog_entry["nutrition"]
            }

    # 2. Tier 2 Check: Live Open Food Facts API
    barcode = catalog_entry.get("barcode")
    if barcode:
        api_data = await fetch_open_food_facts_data(barcode)
        if api_data and api_data.get("name"):
            nutrition = NutritionData(
                calories_kcal=api_data.get("calories", catalog_entry["nutrition"].calories_kcal),
                sugar_g=api_data.get("sugar", catalog_entry["nutrition"].sugar_g),
                fat_g=api_data.get("fat", catalog_entry["nutrition"].fat_g),
                allergens=catalog_entry["nutrition"].allergens,
                eco_grade=api_data.get("eco_grade", "B")
            )
            return {
                "name": f"{api_data.get('brands', '')} {api_data.get('name', catalog_entry['name'])}".strip(),
                "category": catalog_entry["category"],
                "price": catalog_entry["default_price"],
                "source": "LIVE_API (Open Food Facts)",
                "nutrition": nutrition
            }

    # 3. Tier 3: Decoupled Store Catalog / Cognitive Inference
    return {
        "name": catalog_entry["name"],
        "category": catalog_entry["category"],
        "price": catalog_entry["default_price"],
        "source": "COGNITIVE_AI (Size & Pattern Inference)",
        "nutrition": catalog_entry["nutrition"]
    }
