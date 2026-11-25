import cv2
import numpy as np
import os

def generate_samples(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Sample 1: The SIH Presentation Tray (Coke Bottle + Canned Soup + Snack Pack)
    # -------------------------------------------------------------
    tray1 = np.ones((600, 900, 3), dtype=np.uint8) * 248  # Clean off-white checkout counter
    # Add subtle checkerboard or tray border
    cv2.rectangle(tray1, (20, 20), (880, 580), (220, 225, 230), 2)
    cv2.putText(tray1, "RETAIL SMART CHECKOUT CONVEYOR // TRAY #01", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 165, 175), 1)

    # 1. Coke Bottle (Dark Red/Burgundy vertical body + cap)
    cv2.rectangle(tray1, (100, 160), (210, 480), (30, 20, 160), -1)  # Red body
    cv2.rectangle(tray1, (135, 120), (175, 160), (20, 20, 180), -1)  # Neck
    cv2.rectangle(tray1, (140, 100), (170, 120), (200, 200, 200), -1)  # Cap
    cv2.putText(tray1, "COCA-COLA", (110, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(tray1, "MRP Rs 50", (115, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)

    # 2. Canned Food (Red/Gold Heinz Can)
    cv2.rectangle(tray1, (340, 220), (470, 460), (40, 80, 190), -1)  # Red Can
    cv2.ellipse(tray1, (405, 220), (65, 15), 0, 0, 360, (180, 190, 200), -1)  # Top lid
    cv2.ellipse(tray1, (405, 460), (65, 15), 0, 0, 360, (30, 60, 150), -1)   # Bottom
    cv2.putText(tray1, "HEINZ SOUP", (350, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(tray1, "MRP Rs 100", (355, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)

    # 3. Snack Pack (Yellow/Green Puffed Chips Bag)
    cv2.rectangle(tray1, (580, 180), (800, 470), (40, 180, 230), -1)  # Yellow Chips bag
    cv2.putText(tray1, "LAYS CHIPS", (620, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(tray1, "PUFFED SNACK", (620, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    cv2.putText(tray1, "MRP Rs 30", (635, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 180), 2)

    cv2.imwrite(os.path.join(output_dir, "sample_tray_coke_snacks.jpg"), tray1)

    # -------------------------------------------------------------
    # Sample 2: Multi-Size Beverages & Water
    # -------------------------------------------------------------
    tray2 = np.ones((600, 900, 3), dtype=np.uint8) * 250
    cv2.rectangle(tray2, (20, 20), (880, 580), (220, 225, 230), 2)
    cv2.putText(tray2, "BEVERAGE SELECTION TRAY // WATER & JUICE", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 165, 175), 1)

    # Mineral Water (Clear Blue/Cyan bottle)
    cv2.rectangle(tray2, (120, 150), (220, 470), (230, 210, 170), -1)
    cv2.rectangle(tray2, (150, 110), (190, 150), (240, 220, 180), -1)
    cv2.rectangle(tray2, (155, 95), (185, 110), (255, 120, 40), -1)  # Blue cap
    cv2.putText(tray2, "BISLERI 1L", (130, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 60, 20), 1)

    # Real Fruit Juice Box (Orange Carton)
    cv2.rectangle(tray2, (340, 180), (460, 460), (30, 120, 230), -1)  # Orange
    cv2.putText(tray2, "REAL JUICE", (350, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Fresh Fruit Apple
    cv2.circle(tray2, (620, 340), 65, (40, 40, 210), -1)
    cv2.putText(tray2, "APPLE", (595, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Canned Drink
    cv2.rectangle(tray2, (740, 220), (840, 450), (50, 60, 180), -1)
    cv2.putText(tray2, "SODA", (755, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, "sample_tray_beverages.jpg"), tray2)

    # -------------------------------------------------------------
    # Sample 3: Anti-Theft / Occlusion Test (Stacked/Overlapping items)
    # -------------------------------------------------------------
    tray3 = np.ones((600, 900, 3), dtype=np.uint8) * 248
    cv2.rectangle(tray3, (20, 20), (880, 580), (220, 225, 230), 2)
    cv2.putText(tray3, "LOSS PREVENTION BENCHMARK // OVERLAPPING CONCEALMENT", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 165, 175), 1)

    # Item A: Large Snack Pack
    cv2.rectangle(tray3, (250, 170), (520, 470), (40, 190, 240), -1)
    cv2.putText(tray3, "SNACK PACK", (320, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)

    # Item B: Bottle placed DIRECTLY OVER top of snack pack (Triggering IoU Overlap alert!)
    cv2.rectangle(tray3, (380, 190), (490, 490), (30, 20, 160), -1)
    cv2.putText(tray3, "COKE (OVERLAPPING)", (385, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Clean separate item on right
    cv2.rectangle(tray3, (660, 220), (780, 450), (40, 80, 190), -1)
    cv2.putText(tray3, "CANNED FOOD", (670, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, "sample_tray_occlusion.jpg"), tray3)
    print("Samples generated successfully in:", output_dir)

if __name__ == "__main__":
    generate_samples(r"D:\vision-auto-billing\static\samples")
