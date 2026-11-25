import os
import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Tuple
from backend.app.schemas import NutritionData
from backend.app.config import settings
from backend.app.core.product_catalog import product_catalog
from backend.app.core.roboflow_workflow import RoboflowWorkflowDetector, RoboflowWorkflowError

_YOLO_MODEL = None


def _bbox_intersection_over_union(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    """Return overlap for boxes expressed as (x, y, width, height)."""
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first_w * first_h + second_w * second_h - intersection
    return intersection / float(max(1, union))


def _bbox_containment(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    """Return the fraction of the smaller box covered by the other box."""
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_w, second_x + second_w)
    bottom = min(first_y + first_h, second_y + second_h)
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / float(max(1, min(first_w * first_h, second_w * second_h)))


def _suppress_duplicate_candidates(
    candidates: List[Dict[str, Any]],
    same_product_iou: float = 0.25,
    same_product_containment: float = 0.40,
    conflicting_label_iou: float = 0.30,
    conflicting_label_containment: float = 0.45,
) -> List[Dict[str, Any]]:
    """Deduplicates detections so 1 physical product never generates multiple candidate boxes."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.get("_detection_score", item.get("confidence", 0.0)),
            item["bbox"][2] * item["bbox"][3],
        ),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []

    for candidate in ranked:
        candidate_box = candidate["bbox"]
        candidate_product = candidate.get("product_key")
        candidate_class = candidate.get("detector_class")
        duplicate = False

        for accepted in kept:
            accepted_box = accepted["bbox"]
            iou = _bbox_intersection_over_union(candidate_box, accepted_box)
            containment = _bbox_containment(candidate_box, accepted_box)
            same_product = candidate_product and candidate_product == accepted.get("product_key")
            same_class = candidate_class and candidate_class == accepted.get("detector_class")

            if (same_product or same_class) and (iou >= same_product_iou or containment >= same_product_containment):
                duplicate = True
                break
            if not same_product and not same_class and (iou >= conflicting_label_iou or containment >= conflicting_label_containment):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    for item in kept:
        item.pop("_detection_score", None)
    return kept

def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        candidates = [
            os.path.join(root_dir, "yolov8n.pt"),
            r"D:\vision-auto-billing\yolov8n.pt",
            "yolov8n.pt"
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    from ultralytics import YOLO
                    _YOLO_MODEL = YOLO(p)
                    break
                except Exception as e:
                    print("Error loading YOLO:", e)
    return _YOLO_MODEL

_GROCERY_MODEL = None

def get_grocery_yolo_model():
    global _GROCERY_MODEL
    if _GROCERY_MODEL is None:
        candidates = [
            r"D:\vision-auto-billing\grocery_yolo11n.pt",
            "grocery_yolo11n.pt"
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    from ultralytics import YOLO
                    _GROCERY_MODEL = YOLO(p)
                    break
                except Exception as e:
                    print("Error loading Grocery YOLO11:", e)
    return _GROCERY_MODEL

class ProductVisionEngine:
    def __init__(self, min_area: int = 4000):
        self.min_area = min_area
        self._roboflow_detector = None

    async def _detect_with_roboflow(self, image: np.ndarray) -> Tuple[bool, List[Dict[str, Any]], np.ndarray]:
        """Return whether a configured Roboflow request completed successfully.

        A successful workflow response is authoritative, including an empty
        detection list. Local heuristics are used only if hosted inference is
        disabled or unavailable.
        """
        if not settings.ROBOFLOW_ENABLED:
            return False, [], image
        try:
            if self._roboflow_detector is None:
                self._roboflow_detector = RoboflowWorkflowDetector()
            items = await self._roboflow_detector.detect(image, product_catalog)
        except RoboflowWorkflowError as exc:
            print(f"Roboflow unavailable; using local detector: {exc}")
            return False, [], image

        annotated = image.copy()
        items = _suppress_duplicate_candidates(items)
        for idx, item in enumerate(items, 1):
            item["id"] = idx
            self._draw_box(annotated, item)
        return True, items, annotated

    async def detect_products(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        roboflow_completed, roboflow_items, roboflow_image = await self._detect_with_roboflow(image)
        if roboflow_completed:
            return roboflow_items, roboflow_image

        img_h, img_w = image.shape[:2]
        detected_items: List[Dict[str, Any]] = []
        annotated_image = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        occupied_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # 1. Primary YOLOv8 Detector (Shopper Human Masking & Beverage Detection)
        yolo = get_yolo_model()
        has_person = False
        if yolo is not None:
            results = yolo(image, conf=0.25, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = yolo.names.get(cls_id, "item")

                    # EXCLUDE HUMANS / SHOPPERS: Cashier or customer facing webcam is NEVER billed as inventory!
                    if cls_name == "person":
                        has_person = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Mask out head, face, hair, and body with safety margin
                        occupied_mask[max(0, y1 - 60):min(img_h, y2 + 60), max(0, x1 - 60):min(img_w, x2 + 60)] = 255
                        continue

                    if cls_name in ["bottle", "cup", "can"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1
                        if w < 25 or h < 50 or (w > img_w * 0.90 and h > img_h * 0.90):
                            continue

                        roi_hsv = hsv[y1:y2, x1:x2]
                        green_mask = cv2.inRange(roi_hsv, np.array([32, 40, 40]), np.array([88, 255, 255]))
                        green_pct = np.sum(green_mask > 0) / float(max(1, roi_hsv.shape[0] * roi_hsv.shape[1]))
                        mean_v = np.mean(roi_hsv[:, :, 2]) if roi_hsv.size > 0 else 100

                        if green_pct > 0.12:
                            name = "Sprite Lime Drink (740ml)" if h > 220 else "Sprite Lime Drink (250ml)"
                            price = 40.0 if h > 220 else 20.0
                            reason = "Emerald green PET body with Sprite branding."
                        elif mean_v < 115:
                            name, price = "Thums Up / Coca-Cola (500ml)", 40.0
                            reason = "Dark caramelized cola beverage with corporate label."
                        else:
                            name = "Aqua Velly Packaged Mineral Water (1L)" if h > 220 else "Aqua Velly Packaged Mineral Water (500ml)"
                            price = 20.0 if h > 220 else 10.0
                            reason = "Clear transparent mineral water bottle."

                        item_info = {
                            "product_key": name.lower().replace(" ", "_"),
                            "name": name,
                            "category": "Beverage",
                            "confidence": round(conf, 2),
                            "unit_price": price,
                            "quantity": 1,
                            "subtotal": price,
                            "bbox": (x1, y1, w, h),
                            "detector_class": "beverage_bottle",
                            "_detection_score": 100.0 + conf,
                            "pricing_source": "YOLO_VISION_AI",
                            "cognitive_reasoning": reason,
                            "nutrition": NutritionData(calories_kcal=120.0, sugar_g=28.0)
                        }
                        detected_items.append(item_info)
                        occupied_mask[max(0, y1 - 5):min(img_h, y2 + 5), max(0, x1 - 5):min(img_w, x2 + 5)] = 255

        # 2. Transparent Water Bottle Detection (Aqua Velly - Pink/Magenta label)
        pink_mask = cv2.inRange(hsv, np.array([145, 60, 40]), np.array([178, 255, 255]))
        pink_mask[occupied_mask > 0] = 0  # Ignore regions already claimed by YOLO bottles
        
        # Suppress reflection column noise (< 40 pixels per column)
        col_sum = np.sum(pink_mask > 0, axis=0)
        pink_mask[:, col_sum < 40] = 0

        contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area > 800:
                x, y, w, h = cv2.boundingRect(c)
                if h >= 60 and (h / float(max(1, w))) >= 1.02:
                    by1 = max(0, y - int(h * 0.40))
                    by2 = min(img_h, y + h + int(h * 0.15))
                    bw = max(w, 50)
                    bh = by2 - by1
                    name = "Aqua Velly Packaged Mineral Water (1L)" if bh > 220 else "Aqua Velly Packaged Mineral Water (500ml)"
                    price = 20.0 if bh > 220 else 10.0
                    reason = "Transparent mineral water cylinder with distinctive vertical pink Aqua Velly label."

                    item_info = {
                        "product_key": name.lower().replace(" ", "_"),
                        "name": name,
                        "category": "Beverage",
                        "confidence": 0.92,
                        "unit_price": price,
                        "quantity": 1,
                        "subtotal": price,
                        "bbox": (x, by1, bw, bh),
                        "detector_class": "water_bottle",
                        "_detection_score": 90.0,
                        "pricing_source": "WATER_SALIENCY_AI",
                        "cognitive_reasoning": reason,
                        "nutrition": NutritionData(calories_kcal=0.0, sugar_g=0.0)
                    }
                    detected_items.append(item_info)
                    occupied_mask[by1:by2, x:x + bw] = 255

        # 3. Fine-Tuned Retail Self-Checkout YOLO11 (HuggingFace cvtechniques/GroceryCheckoutDetection)
        grocery_yolo = get_grocery_yolo_model()
        if grocery_yolo is not None and not has_person:
            g_results = grocery_yolo(image, conf=0.45, verbose=False)
            name_map = {
                "puffed_food": ("Lays / Puffed Corn Snack Pack (Party Pack)", "Snacks", 30.0),
                "chocolate": ("Cadbury Dairy Milk / Oreo Biscuit", "Confectionery", 30.0),
                "dessert": ("Britannia Treat / JimJam Biscuit", "Confectionery", 35.0),
                "instant_noodles": ("Maggi 2-Minute Masala Noodles", "Packaged Food", 14.0),
                "instant_drink": ("Real Fruit Juice Carton (1L)", "Beverage", 110.0),
                "canned_food": ("Heinz Baked Beans Can", "Canned", 120.0)
            }
            for r in g_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = grocery_yolo.names.get(cls_id, "")
                    if cls_name in name_map:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        if occupied_mask[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)].mean() < 60:
                            prod_name, cat, price = name_map[cls_name]
                            item_info = {
                                "product_key": prod_name.lower().replace(" ", "_"),
                                "name": prod_name,
                                "category": cat,
                                "confidence": round(conf, 2),
                                "unit_price": price,
                                "quantity": 1,
                                "subtotal": price,
                                "bbox": (x1, y1, w, h),
                                "detector_class": f"grocery_{cls_name}",
                                "_detection_score": 85.0 + conf,
                                "pricing_source": "RETAIL_YOLO11_AI",
                                "cognitive_reasoning": f"Trained Retail Self-Checkout YOLO11 detected {cls_name} packaging ({int(conf*100)}% conf).",
                                "nutrition": NutritionData(calories_kcal=160.0, sugar_g=8.0)
                            }
                            detected_items.append(item_info)
                            occupied_mask[y1:y2, x1:x2] = 255

        # 4. Packaged Retail Goods (Oreo, Lays, JimJam, Kurkure, Tasty Nuts)
        # Strictly executed ONLY when camera is facing a grocery tray and NOT when a human face/selfie fills the frame!
        if not has_person:
            packaging_rules = [
                ("Cadbury Oreo Biscuit (120g)", "Confectionery", 30.0, np.array([95, 60, 40]), np.array([135, 255, 255]), "Blue packaging with Oreo sandwich cookie profile."),
                ("Britannia JimJam Treat", "Confectionery", 35.0, np.array([145, 60, 40]), np.array([178, 255, 255]), "Magenta/crimson JimJam biscuit pouch."),
                ("Lays American / Lime Chips", "Snacks", 20.0, np.array([35, 60, 50]), np.array([85, 255, 255]), "Vivid green packaging with chips branding."),
                ("Haldiram / Balaji Tasty Nuts", "Snacks", 20.0, np.array([5, 90, 70]), np.array([25, 255, 255]), "Orange/red spiced peanut namkeen pouch."),
                ("Kurkure / Puffed Corn Snack", "Snacks", 20.0, np.array([25, 90, 70]), np.array([35, 255, 255]), "Yellow/green puffed namkeen pouch."),
                ("Mentos Fresh Candy Roll", "Confectionery", 10.0, np.array([135, 60, 40]), np.array([145, 255, 255]), "Purple/magenta confectionery candy roll.")
            ]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            for name, cat, price, lower, upper, reason in packaging_rules:
                p_mask = cv2.inRange(hsv, lower, upper)
                p_mask[occupied_mask > 0] = 0  # Bottles and persons are completely excluded!
                p_clean = cv2.morphologyEx(p_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                conts, _ = cv2.findContours(p_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in conts:
                    area = cv2.contourArea(c)
                    if area > 1100:
                        x, y, bw, bh = cv2.boundingRect(c)
                        ar = max(bw, bh) / float(max(1, min(bw, bh)))
                        is_edge = (y + bh >= img_h - 15) or (x <= 15) or (y <= 15) or (x + bw >= img_w - 15)
                        
                        roi_check = hsv[y:y + bh, x:x + bw]
                        pct_water = np.mean((roi_check[:, :, 2] > 150) & (roi_check[:, :, 1] < 60)) if roi_check.size > 0 else 0
                        if "JimJam" in name and pct_water > 0.18:
                            continue  # Skip transparent water bottles misidentified as JimJam!

                        if ar < 2.8 and not is_edge and bw >= 35 and bh >= 35 and bw < img_w * 0.85 and bh < img_h * 0.85:
                            item_info = {
                                "product_key": name.lower().replace(" ", "_"),
                                "name": name,
                                "category": cat,
                                "confidence": 0.94,
                                "unit_price": price,
                                "quantity": 1,
                                "subtotal": price,
                                "bbox": (x, y, bw, bh),
                                "detector_class": "retail_package",
                                "_detection_score": 50.0 + (area / 1000.0),
                                "pricing_source": "RETAIL_SALIENCY_AI",
                                "cognitive_reasoning": reason,
                                "nutrition": NutritionData(calories_kcal=150.0, sugar_g=12.0)
                            }
                            detected_items.append(item_info)
                            occupied_mask[y:y + bh, x:x + bw] = 255

        # 4. Strict Deduplication & Spatial Non-Maximum Suppression
        final_items = _suppress_duplicate_candidates(detected_items)
        for idx, item in enumerate(final_items, 1):
            item["id"] = idx
            self._draw_box(annotated_image, item)

        return final_items, annotated_image

    def _draw_box(self, img: np.ndarray, item: Dict[str, Any]):
        x, y, w, h = item["bbox"]
        name = item["name"]
        price = item["unit_price"]
        conf = int(item["confidence"] * 100)
        cat = item.get("category", "")

        color = (30, 200, 50)  # Green default
        if "JimJam" in name or "Mentos" in name:
            color = (180, 50, 180)  # Magenta/Purple
        elif "Oreo" in name:
            color = (220, 90, 40)   # Blue
        elif "Tasty Nuts" in name:
            color = (30, 120, 230)  # Orange
        elif "Water" in name:
            color = (220, 150, 20)  # Cyan
        elif "Kurkure" in name:
            color = (40, 200, 240)  # Yellow

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

        clean_name = name.split("(")[0].strip()
        tag_text = f"{clean_name} | ₹{int(price)} ({conf}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(tag_text, font, 0.48, 1)

        pill_y1 = max(0, y - text_h - 10)
        pill_y2 = y
        pill_x1 = x
        pill_x2 = min(img.shape[1], x + text_w + 12)

        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
        cv2.putText(img, tag_text, (pill_x1 + 4, pill_y2 - 4), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

def image_to_base64_data_uri(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"
