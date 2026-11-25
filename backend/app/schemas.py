from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

class NutritionData(BaseModel):
    calories_kcal: Optional[float] = 0.0
    sugar_g: Optional[float] = 0.0
    fat_g: Optional[float] = 0.0
    allergens: List[str] = Field(default_factory=list)
    eco_grade: Optional[str] = "B"

class DetectedItem(BaseModel):
    id: int
    name: str
    category: str
    confidence: float
    unit_price: float
    quantity: int = 1
    subtotal: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    pricing_source: str = "COGNITIVE_AI"  # OCR_PACKAGING | LIVE_API | COGNITIVE_AI | STORE_CATALOG
    cognitive_reasoning: str = ""
    nutrition: Optional[NutritionData] = None
    sku: Optional[str] = None
    suggested_name: Optional[str] = None
    recognition_status: str = "REVIEW_REQUIRED"  # BILLABLE | REVIEW_REQUIRED | UNKNOWN | OCCLUDED
    billable: bool = False
    evidence: List[str] = Field(default_factory=list)

class AntiTheftReport(BaseModel):
    status: str = "SECURE"  # SECURE | WARNING | CRITICAL
    occlusion_detected: bool = False
    message: str = "All scanned items cleanly visible on tray."
    flagged_pairs: List[str] = Field(default_factory=list)
    tray_density_score: float = 0.15

class CheckoutReceipt(BaseModel):
    invoice_number: str
    timestamp: str
    items: List[DetectedItem]
    total_quantity: int
    billable_quantity: int
    review_required_quantity: int
    subtotal: float
    cgst: float
    sgst: float
    total_tax: float
    grand_total: float
    estimated_subtotal: float
    estimated_total_tax: float
    estimated_grand_total: float
    anti_theft: AntiTheftReport
    nutrition_summary: Dict[str, Any]
    processing_time_ms: float

class DetectionResponse(BaseModel):
    success: bool
    annotated_image: str  # Base64 data URI
    receipt: CheckoutReceipt

class InventoryUpdate(BaseModel):
    product_key: str
    name: str
    category: str
    price: float

class CashierChatRequest(BaseModel):
    message: str
    items: List[Dict[str, Any]] = Field(default_factory=list)
    anti_theft_status: str = "SECURE"
    grand_total: float = 0.0

class CashierChatResponse(BaseModel):
    reply: str
    provider: str
