import os
import cv2
import numpy as np
from ultralytics import YOLO

def test_pipeline(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Could not load image:", img_path)
        return
    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    yolo = YOLO(r"D:\vision-auto-billing\yolov8n.pt")
    res = yolo(img, conf=0.25, verbose=False)
    
    detected = []
    occupied_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # 1. YOLO Beverage/Bottle pass
    for r in res:
        for b in r.boxes:
            cls_name = yolo.names.get(int(b.cls[0]))
            conf = float(b.conf[0])
            if cls_name in ["bottle", "cup", "can"]:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w < 25 or h < 50 or (w > img_w * 0.90 and h > img_h * 0.90):
                    continue
                roi = hsv[y1:y2, x1:x2]
                green_m = cv2.inRange(roi, np.array([32, 40, 40]), np.array([88, 255, 255]))
                green_pct = np.sum(green_m > 0) / float(max(1, roi.shape[0] * roi.shape[1]))
                mean_v = np.mean(roi[:, :, 2])
                
                if green_pct > 0.12:
                    name = "Sprite Lime Drink (740ml)" if h > 220 else "Sprite Lime Drink (250ml)"
                    price = 40.0 if h > 220 else 20.0
                elif mean_v < 115:
                    name, price = "Thums Up / Coca-Cola (500ml)", 40.0
                else:
                    name = "Aqua Velly Packaged Mineral Water (1L)" if h > 220 else "Aqua Velly Packaged Mineral Water (500ml)"
                    price = 20.0 if h > 220 else 10.0
                    
                detected.append({"name": name, "price": price, "bbox": (x1, y1, w, h), "conf": conf, "source": "YOLO"})
                occupied_mask[max(0, y1-5):min(img_h, y2+5), max(0, x1-5):min(img_w, x2+5)] = 255
                
    # 2. Transparent Water Bottle pass (Aqua Velly)
    pink_mask = cv2.inRange(hsv, np.array([145, 50, 40]), np.array([178, 255, 255]))
    pink_mask[occupied_mask > 0] = 0  # Ignore regions already occupied by YOLO bottles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean_pink = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(clean_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1800:
            x, y, w, h = cv2.boundingRect(c)
            if h / float(max(1, w)) >= 1.05 and h >= 70:
                by1 = max(0, y - int(h * 0.40))
                by2 = min(img_h, y + h + int(h * 0.15))
                bx1 = max(0, x - int(w * 0.20))
                bx2 = min(img_w, x + w + int(w * 0.20))
                bw, bh = bx2 - bx1, by2 - by1
                name = "Aqua Velly Packaged Mineral Water (1L)" if bh > 220 else "Aqua Velly Packaged Mineral Water (500ml)"
                price = 20.0 if bh > 220 else 10.0
                detected.append({"name": name, "price": price, "bbox": (bx1, by1, bw, bh), "conf": 0.92, "source": "WATER_AI"})
                occupied_mask[by1:by2, bx1:bx2] = 255

    # 3. Packaged retail goods (ONLY where occupied_mask == 0)
    rules = [
        ("Cadbury Oreo Biscuit (120g)", "Confectionery", 30.0, np.array([95, 60, 40]), np.array([135, 255, 255])),
        ("Britannia JimJam Treat", "Confectionery", 35.0, np.array([145, 60, 40]), np.array([178, 255, 255])),
        ("Lays American / Lime Chips", "Snacks", 20.0, np.array([35, 60, 50]), np.array([85, 255, 255])),
        ("Haldiram / Balaji Tasty Nuts", "Snacks", 20.0, np.array([5, 90, 70]), np.array([25, 255, 255])),
        ("Kurkure / Puffed Corn Snack", "Snacks", 20.0, np.array([25, 90, 70]), np.array([35, 255, 255]))
    ]
    
    for name, cat, price, lower, upper in rules:
        p_mask = cv2.inRange(hsv, lower, upper)
        p_mask[occupied_mask > 0] = 0  # Bottle regions are strictly excluded!
        p_clean = cv2.morphologyEx(p_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        conts, _ = cv2.findContours(p_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in conts:
            area = cv2.contourArea(c)
            if area > 1400:
                x, y, w, h = cv2.boundingRect(c)
                ar = max(w, h) / float(max(1, min(w, h)))
                is_edge = (y + h >= img_h - 15) or (x <= 5) or (y <= 5) or (x + w >= img_w - 5)
                if ar < 2.5 and not is_edge and w >= 40 and h >= 40:
                    detected.append({"name": name, "price": price, "bbox": (x, y, w, h), "conf": 0.94, "source": "RETAIL_PACK"})
                    occupied_mask[y:y+h, x:x+w] = 255

    print(f"=== Results for {os.path.basename(img_path)} ===")
    print(f"Total detected: {len(detected)}")
    for d in detected:
        print(f"  - {d['name']} ({d['source']}) bbox={d['bbox']}")

if __name__ == "__main__":
    for p in [
        r"C:\Users\Shree\.gemini\antigravity\brain\f1dc1b8c-a5cf-4836-823b-1467754a5b20\.user_uploaded\media_1795573211016_25fbba8b.jpg",
        r"C:\Users\Shree\.gemini\antigravity\brain\f1dc1b8c-a5cf-4836-823b-1467754a5b20\.user_uploaded\media_1795445982729_f2754801.jpg"
    ]:
        test_pipeline(p)
