import time
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.schemas import (
    DetectionResponse, CheckoutReceipt, AntiTheftReport, 
    DetectedItem, InventoryUpdate, CashierChatRequest, CashierChatResponse
)
from backend.app.core.detector import ProductVisionEngine, image_to_base64_data_uri
from backend.app.core.anti_theft import analyze_tray_loss_prevention
from backend.app.core.product_catalog import product_catalog
from backend.app.core.ai_cashier import ai_cashier

app = FastAPI(
    title=settings.APP_NAME,
    description="VisionCheckout: Autonomous AI Smart Checkout & Loss Prevention System with Real-Time Dynamic Pricing",
    version=settings.VERSION
)

# Enable CORS for local testing and Vercel/Render frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vision Engine
vision_engine = ProductVisionEngine(min_area=settings.MIN_CONTOUR_AREA)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

def build_checkout_receipt(
    items: List[Dict[str, Any]], 
    anti_theft: AntiTheftReport, 
    processing_time_ms: float
) -> CheckoutReceipt:
    """Builds a comprehensive POS retail invoice with tax calculation and nutritional summaries."""
    billable_items = [item for item in items if item.get("billable", False)]
    estimated_items = [item for item in items if item.get("unit_price", 0) > 0]
    subtotal = sum(item["unit_price"] * item["quantity"] for item in billable_items)
    cgst = round(subtotal * (settings.GST_RATE / 2), 2)
    sgst = round(subtotal * (settings.GST_RATE / 2), 2)
    total_tax = round(cgst + sgst, 2)
    grand_total = round(subtotal + total_tax, 2)
    estimated_subtotal = round(sum(item["unit_price"] * item["quantity"] for item in estimated_items), 2)
    estimated_total_tax = round(estimated_subtotal * settings.GST_RATE, 2)
    estimated_grand_total = round(estimated_subtotal + estimated_total_tax, 2)
    
    # Nutrition aggregation
    total_calories = 0.0
    total_sugar = 0.0
    allergens = set()
    for item in estimated_items:
        nutr = item.get("nutrition")
        if nutr:
            if isinstance(nutr, dict):
                total_calories += nutr.get("calories_kcal", 0.0) or 0.0
                total_sugar += nutr.get("sugar_g", 0.0) or 0.0
                if nutr.get("allergens"):
                    allergens.update(nutr["allergens"])
            else:
                total_calories += getattr(nutr, "calories_kcal", 0.0) or 0.0
                total_sugar += getattr(nutr, "sugar_g", 0.0) or 0.0
                if getattr(nutr, "allergens", None):
                    allergens.update(nutr.allergens)

    nutrition_summary = {
        "total_calories_kcal": round(total_calories, 1),
        "total_sugar_g": round(total_sugar, 1),
        "allergens_flagged": list(allergens),
        "health_rating": "Moderate" if total_sugar < 60 else "High Sugar Alert"
    }

    invoice_no = f"INV-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    pydantic_items = [DetectedItem(**item) for item in items]

    return CheckoutReceipt(
        invoice_number=invoice_no,
        timestamp=datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
        items=pydantic_items,
        total_quantity=sum(item["quantity"] for item in items),
        billable_quantity=sum(item["quantity"] for item in billable_items),
        review_required_quantity=sum(item["quantity"] for item in items if not item.get("billable", False)),
        subtotal=round(subtotal, 2),
        cgst=cgst,
        sgst=sgst,
        total_tax=total_tax,
        grand_total=grand_total,
        estimated_subtotal=estimated_subtotal,
        estimated_total_tax=estimated_total_tax,
        estimated_grand_total=estimated_grand_total,
        anti_theft=anti_theft,
        nutrition_summary=nutrition_summary,
        processing_time_ms=round(processing_time_ms, 2)
    )


def apply_checkout_guardrails(items: List[Dict[str, Any]], anti_theft: AntiTheftReport) -> List[Dict[str, Any]]:
    """Prevent a generic visual detection from becoming an unverified customer charge.

    Items with >= 65% confidence or Roboflow exact-SKU matches are verified and billable
    unless the tray fails the anti-theft occlusion check. Low-confidence candidates require review.
    """
    for item in items:
        suggested_name = item["name"]
        item["suggested_name"] = suggested_name
        item["subtotal"] = round(item["unit_price"] * item["quantity"], 2)
        mapped_sku = item.pop("roboflow_sku", None)
        roboflow_label = item.pop("roboflow_label", None)
        exact_sku_match = bool(mapped_sku and item.get("pricing_source") == "ROBOFLOW_STORE_CATALOG")
        confidence = float(item.get("confidence", 0.0))

        if anti_theft.occlusion_detected:
            item["sku"] = mapped_sku if exact_sku_match else None
            item["billable"] = False
            item["recognition_status"] = "OCCLUDED"
            item["evidence"] = ["Object location detected by vision model.", "Items overlap; separate them and rescan before payment."]
            item["cognitive_reasoning"] = (
                f"Candidate: {suggested_name}. Items overlap, so this item must be separated and rescanned."
            )
        elif exact_sku_match:
            item["sku"] = mapped_sku
            item["billable"] = True
            item["recognition_status"] = "BILLABLE"
            item["evidence"] = [
                f"Roboflow exact-SKU class '{roboflow_label}' is explicitly mapped to store SKU '{mapped_sku}'.",
                "Price was resolved from the store catalogue.",
            ]
            item["cognitive_reasoning"] = f"Verified SKU {mapped_sku} from configured Roboflow product model."
        elif confidence >= 0.65:
            # Autonomous AI smart-checkout verification
            item["sku"] = item.get("product_key", "sku_auto")
            item["billable"] = True
            item["recognition_status"] = "BILLABLE"
            source = item.get("pricing_source", "VISION_AI")
            item["evidence"] = [
                f"Vision AI ({source}) verified {suggested_name} at {int(confidence * 100)}% confidence.",
                "Price and SKU resolved from store product catalog.",
            ]
            if not item.get("cognitive_reasoning"):
                item["cognitive_reasoning"] = f"Autonomous checkout verified {suggested_name} (₹{item['unit_price']})."
        else:
            item["sku"] = None
            item["billable"] = False
            item["recognition_status"] = "REVIEW_REQUIRED"
            item["evidence"] = [
                "Object location detected by vision model.",
                "Confidence below autonomous billing threshold (65%); requires manual verification.",
            ]
            item["cognitive_reasoning"] = (
                f"Candidate: {suggested_name}. Product confidence ({int(confidence * 100)}%) requires manual confirmation."
            )
    return items

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "models": {
            "vision": "Generic object detection; SKU auto-billing disabled until calibrated recognition is added",
            "loss_prevention": "IoU Overlap Anomaly Detector",
            "pricing": "Persistent store product master; unverified detections require review",
            "ai_agent": "Aura Smart Retail Cashier & Nutritionist"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/samples")
def list_sample_trays():
    """Returns metadata for 1-click recruiter demo trays."""
    return [
        {
            "id": "coke_snacks",
            "title": "SIH Hackathon Tray (Coke + Cans + Snacks)",
            "description": "Matches Team Thunder Forge SIH Presentation. Multi-product beverage & snack combo.",
            "file": "sample_tray_coke_snacks.jpg"
        },
        {
            "id": "beverages",
            "title": "Beverage & Fruit Selection",
            "description": "Mineral Water, Real Fruit Juice carton, Fresh Apple, and Soda Can.",
            "file": "sample_tray_beverages.jpg"
        },
        {
            "id": "occlusion",
            "title": "Anti-Theft / Loss Prevention Test",
            "description": "Items placed directly overlapping to trigger the AI spatial occlusion alert.",
            "file": "sample_tray_occlusion.jpg"
        }
    ]

@app.post("/api/v1/detect-sample/{sample_id}", response_model=DetectionResponse)
async def detect_sample_tray(sample_id: str):
    """Executes detection on a pre-built sample image with 1-click."""
    file_map = {
        "coke_snacks": "sample_tray_coke_snacks.jpg",
        "beverages": "sample_tray_beverages.jpg",
        "occlusion": "sample_tray_occlusion.jpg"
    }
    filename = file_map.get(sample_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Sample tray not found")

    filepath = os.path.join(SAMPLES_DIR, filename)
    image = cv2.imread(filepath)
    if image is None:
        raise HTTPException(status_code=500, detail="Sample image file missing on server")

    start_t = time.perf_counter()
    detected_items, annotated_img = await vision_engine.detect_products(image)
    anti_theft = analyze_tray_loss_prevention(detected_items, image.shape, threshold=settings.IOU_OCCLUSION_THRESHOLD)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    guarded_items = apply_checkout_guardrails(detected_items, anti_theft)
    receipt = build_checkout_receipt(guarded_items, anti_theft, elapsed_ms)
    b64_image = image_to_base64_data_uri(annotated_img)

    return DetectionResponse(
        success=True,
        annotated_image=b64_image,
        receipt=receipt
    )

@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_uploaded_image(file: UploadFile = File(...)):
    """Receives user-uploaded image or camera snapshot, runs detection & real-time billing."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image uploaded. Could not decode.")

    start_t = time.perf_counter()
    detected_items, annotated_img = await vision_engine.detect_products(image)
    anti_theft = analyze_tray_loss_prevention(detected_items, image.shape, threshold=settings.IOU_OCCLUSION_THRESHOLD)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    guarded_items = apply_checkout_guardrails(detected_items, anti_theft)
    receipt = build_checkout_receipt(guarded_items, anti_theft, elapsed_ms)
    b64_image = image_to_base64_data_uri(annotated_img)

    return DetectionResponse(
        success=True,
        annotated_image=b64_image,
        receipt=receipt
    )

@app.post("/api/v1/chat-cashier", response_model=CashierChatResponse)
async def chat_with_cashier(req: CashierChatRequest):
    """Interactive AI Smart Cashier & Retail Concierge conversation."""
    result = await ai_cashier.chat(
        user_message=req.message,
        current_items=req.items,
        anti_theft_status=req.anti_theft_status,
        grand_total=req.grand_total
    )
    return CashierChatResponse(reply=result["reply"], provider=result["provider"])

@app.get("/api/v1/inventory")
def get_inventory():
    """Returns the persistent store-owned product master."""
    return product_catalog.list_products()

@app.post("/api/v1/inventory/update")
def update_product_price(payload: InventoryUpdate):
    """Updates a durable catalogue price; recognition must still verify the SKU."""
    product = product_catalog.upsert_product(
        payload.product_key, payload.name, payload.category, payload.price
    )
    return {"message": f"Updated {payload.name} price to ₹{payload.price}", "product": product}

# Mount static folder
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "VisionCheckout API is running. Visit /docs for Swagger."}
