import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "VisionCheckout AI"
    VERSION: str = "2.1.0"
    API_PREFIX: str = "/api/v1"
    
    # Computer Vision Settings
    MIN_CONTOUR_AREA: int = 4000
    CONFIDENCE_THRESHOLD: float = 0.50
    AUTO_BILL_CONFIDENCE: float = 0.98
    AUTO_BILL_MARGIN: float = 0.10
    IOU_OCCLUSION_THRESHOLD: float = 0.35  # Overlap above this triggers theft/obscured alert

    # Product master. Use a managed database URL in production; this SQLite file
    # provides a durable local development catalogue.
    CATALOG_DATABASE_PATH: str = "data/visioncheckout_catalog.db"
    
    # Tax Settings
    GST_RATE: float = 0.05  # 5% GST for retail groceries/beverages
    
    # Currency
    CURRENCY_SYMBOL: str = "₹"
    CURRENCY_CODE: str = "INR"

    # Optional hosted exact-SKU detector. Enable it only after mapping every
    # Roboflow class label to a SKU in the store catalogue.
    ROBOFLOW_ENABLED: bool = False
    ROBOFLOW_API_KEY: str = ""
    ROBOFLOW_WORKSPACE: str = "pushkar-patade"
    ROBOFLOW_WORKFLOW_ID: str = "grocery-auto-billing-demo-1795578899600"
    ROBOFLOW_CONFIDENCE_THRESHOLD: float = 0.80
    # JSON object, e.g. {"coke_500ml":"coke_bottle"}.
    ROBOFLOW_LABEL_TO_SKU_JSON: str = "{}"
    
settings = Settings()
