import os
import base64
import json
import cv2
import numpy as np
import httpx
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env"))
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

async def detect_with_gemini_vision(image: np.ndarray) -> Optional[Tuple[List[Dict[str, Any]], np.ndarray]]:
    if not API_KEY:
        return None

    img_h, img_w = image.shape[:2]
    
    # Resize to max 480px for ultra-fast <500ms transmission
    scale = min(480.0 / max(img_h, img_w), 1.0)
    small_h = int(img_h * scale)
    small_w = int(img_w * scale)
    small = cv2.resize(image, (small_w, small_h))

    success, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not success:
        return None
    b64_image = base64.b64encode(buffer).decode("utf-8")

    prompt = """You are an AI Retail POS Scanner.
Identify every packaged food, beverage, snack, chips, biscuit, candy, or grocery item on the table.
Return ONLY a valid JSON array of objects with:
- name: Exact product name (e.g. Cadbury Oreo, Britannia JimJam, Lays Chips, Haldiram Tasty Nuts, Thums Up)
- category: Beverage | Snacks | Confectionery | Packaged Food | Water
- estimated_price_inr: Realistic Indian retail price in INR (e.g. 10, 20, 30, 40)
- ymin: Normalized top (0 to 1000)
- xmin: Normalized left (0 to 1000)
- ymax: Normalized bottom (0 to 1000)
- xmax: Normalized right (0 to 1000)
- cognitive_reasoning: 1 sentence explaining brand packaging, graphics, and size reasoning

Output ONLY JSON."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2
        }
    }

    try:
        # Strict 3.0s timeout: POS must never freeze!
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.post(url, json=payload)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    raw_text = candidates[0]["content"]["parts"][0]["text"].strip()
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("```")[1]
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:]
                    raw_text = raw_text.strip()
                    items_json = json.loads(raw_text)
                    if isinstance(items_json, list) and len(items_json) > 0:
                        detected_items = []
                        annotated_img = image.copy()
                        counter = 1
                        for it in items_json:
                            name = it.get("name", "Packaged Item")
                            price = float(it.get("estimated_price_inr", 20.0))
                            ymin = int((it.get("ymin", 0) / 1000.0) * img_h)
                            xmin = int((it.get("xmin", 0) / 1000.0) * img_w)
                            ymax = int((it.get("ymax", 1000) / 1000.0) * img_h)
                            xmax = int((it.get("xmax", 1000) / 1000.0) * img_w)

                            x = max(0, min(img_w - 5, xmin))
                            y = max(0, min(img_h - 5, ymin))
                            w = max(10, min(img_w - x, xmax - xmin))
                            h = max(15, min(img_h - y, ymax - ymin))

                            item_obj = {
                                "id": counter,
                                "product_key": name.lower().replace(" ", "_"),
                                "name": name,
                                "category": it.get("category", "Snacks"),
                                "confidence": 0.98,
                                "unit_price": price,
                                "quantity": 1,
                                "subtotal": price,
                                "bbox": (x, y, w, h),
                                "pricing_source": "MULTIMODAL_VISION_AI (Gemini)",
                                "cognitive_reasoning": it.get("cognitive_reasoning", "Identified via Multimodal Vision AI."),
                                "nutrition": None
                            }
                            detected_items.append(item_obj)
                            _draw_box(annotated_img, item_obj)
                            counter += 1
                        return detected_items, annotated_img
    except Exception:
        pass

    return None

def _draw_box(img: np.ndarray, item: Dict[str, Any]):
    x, y, w, h = item["bbox"]
    name = item["name"]
    price = item["unit_price"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 200, 50), 2, cv2.LINE_AA)

    tag_text = f"{name} | ₹{int(price)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    (text_w, text_h), _ = cv2.getTextSize(tag_text, font, font_scale, 1)

    pill_y1 = max(0, y - text_h - 12)
    pill_y2 = y
    pill_x1 = x
    pill_x2 = min(img.shape[1], x + text_w + 14)

    cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), (30, 200, 50), -1)
    cv2.putText(img, tag_text, (pill_x1 + 6, pill_y2 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
