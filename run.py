import uvicorn
import os
import sys

# Ensure root directory is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 65)
    print("⚡ VisionCheckout AI: Automated Billing & Loss Prevention POS ⚡")
    print("=" * 65)
    print("📍 Local Web App:     http://localhost:8000")
    print("📍 Swagger API Docs:  http://localhost:8000/docs")
    print("📍 Loss Prevention:   IoU Overlap Active (Threshold: 0.35)")
    print("📍 Pricing Engine:    3-Tier (Visual OCR + Live Food Facts API + Catalog)")
    print("📍 AI Retail Agent:   Aura (Groq / Gemini / Local Engine)")
    print("=" * 65)
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
