# ⚡ VisionCheckout AI — Autonomous Retail Billing & Loss Prevention

> **Smart India Hackathon (SIH) Finalist Project** | Team Thunder Forge (PS #SIH25135)  
> *Autonomous Multi-Product Checkout, 3-Tier Real-Time Dynamic Pricing, AI Loss Prevention, and Conversational Retail Agent.*

![Status](https://img.shields.io/badge/Status-Prototype-orange) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-blue) ![ComputerVision](https://img.shields.io/badge/AI-OpenCV_%26_YOLO-purple) ![License](https://img.shields.io/badge/License-MIT-gray)

---

## 📌 Executive Summary

Traditional retail self-checkouts require manual barcode alignment, slowing checkout throughput and causing shrinkage (unscanned or stacked items). **VisionCheckout AI** is a prototype for camera-assisted checkout: it detects tray objects, presents product candidates for review, keeps a persistent local product master, and provides an interactive voice-enabled AI retail concierge ("Aura"). Exact SKU recognition is required before this can replace barcode checkout.

> **Checkout safety:** generic visual detection is not product identification. The application blocks payment for `REVIEW_REQUIRED`, `UNKNOWN`, and `OCCLUDED` candidates. See [PRODUCT_READINESS_ROADMAP.md](PRODUCT_READINESS_ROADMAP.md) for the path to a production system.

---

## 🏛️ System Architecture

```
[ Visual Scanning Tray / Camera Feed ]
                  │
                  ▼
   ┌───────────────────────────────┐
   │  Product Vision Engine        │
   │  (YOLOv8 + Adaptive Contours) │
   └──────────────┬────────────────┘
                  │ Bounding Boxes & Classes
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │             3-Tier Real-Time Pricing Resolver                │
   │  ├─ Tier 1: Visual OCR MRP Extractor (Zero-DB from packaging)│
   │  ├─ Tier 2: Live Open Food Facts API (3M+ Global FMCG items) │
   │  └─ Tier 3: Decoupled Store ERP REST API Catalog             │
   └──────────────┬───────────────────────────────────────────────┘
                  │
                  ├──► [ AI Loss Prevention / Spatial IoU Overlap Engine ]
                  │    (Detects hidden, obscured, or stacked items)
                  │
                  ├──► [ Automated POS Invoice & GST Tax Engine ]
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │          Aura: AI Smart Cashier & Retail Concierge           │
   │  (Powered by Groq Llama-3.3-70B / Gemini / Speech Synthesis) │
   │  - Nutritional Breakdown (Calories, Sugar, Allergens)        │
   │  - Dynamic Combo Deals & Voice Guidance                      │
   └──────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

1. **Multi-Product Candidate Detection:** Locates some bottles, cans, snack bags, boxes, and produce using a generic vision model and OpenCV rules.
2. **Persistent Product Master:** A SQLite-backed SKU catalogue is exposed through `/api/v1/inventory`; it is the source of product pricing, not visual inference.
3. **Safe Billing Gate:** Product candidates are marked `REVIEW_REQUIRED` unless an exact, calibrated SKU matcher supplies evidence sufficient for automatic billing.
4. **Occlusion Warning:** Pairwise box overlap asks for a rescan when items overlap. This is a scanning-quality signal, not proof of theft.
5. **"Aura" AI Retail Concierge (Human-like Interaction):**
   - Answers questions about calories, ingredients, sugar warnings, and combo offers.
   - Speaks aloud using the Web Speech API (Text-to-Speech).
6. **1-Click Prototype Demonstration:**
   - Built-in pre-packaged test trays (including the exact SIH Coke + Snack Pack presentation benchmark) for zero-setup 1-click evaluation.

---

## 🚀 Quickstart (Run Locally)

### 1. Prerequisites
- Python 3.10+
- pip

### 2. Install Dependencies
```bash
git clone https://github.com/Pushkarpatade/vision-auto-billing.git
cd vision-auto-billing
pip install -r requirements.txt
```

### 3. (Optional) Configure hosted SKU recognition and Aura
Copy `.env.example` to `.env`:
```bash
ROBOFLOW_ENABLED=true
ROBOFLOW_API_KEY=your_roboflow_private_api_key
ROBOFLOW_WORKSPACE=pushkar-patade
ROBOFLOW_WORKFLOW_ID=grocery-auto-billing-demo-1795578899600
# Map each exact class label emitted by the workflow to a store SKU.
ROBOFLOW_LABEL_TO_SKU_JSON={"coke_500ml":"coke_bottle"}
GROQ_API_KEY=your_groq_api_key_here      # Free at https://console.groq.com
# Or:
GEMINI_API_KEY=your_gemini_api_key_here  # Free at https://aistudio.google.com
```
Roboflow runs only on the server and uses the configured workflow. A label is billable only when it is explicitly mapped to a SKU in `/api/v1/inventory`, meets the confidence threshold, and the tray has no occlusion warning. Without Roboflow configuration, VisionCheckout automatically uses its built-in local detection as review-only candidates.

### 4. Start the Application
```bash
python run.py
```
- Open **http://localhost:8000** in your browser.
- Interactive Swagger API docs: **http://localhost:8000/docs**.

---

## ☁️ Deployment on Render (100% Free Tier)

1. Push this folder to a GitHub repository (`https://github.com/Pushkarpatade/vision-auto-billing`).
2. Log into [Render.com](https://render.com) and click **New +** ➔ **Web Service**.
3. Select your repository.
4. Configure:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy Web Service** — your live checkout URL will be active within 2 minutes!

For hosted exact-SKU recognition, add `ROBOFLOW_API_KEY` as a Render secret, then add `ROBOFLOW_ENABLED=true`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_WORKFLOW_ID`, and `ROBOFLOW_LABEL_TO_SKU_JSON` in Render's environment settings. Never put the Roboflow key in the frontend or commit it to Git.

---

## 👥 Authors
- **Pushkar Patade** — *Computer Science & Cybersecurity Engineer, MIT ADT University*
- *Team Thunder Forge — Smart India Hackathon*
