from typing import List, Dict, Any, Tuple
from backend.app.schemas import AntiTheftReport

def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes: (x, y, w, h).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)

    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def analyze_tray_loss_prevention(detected_items: List[Dict[str, Any]], image_shape: Tuple[int, int], threshold: float = 0.35) -> AntiTheftReport:
    """
    Analyzes item bounding boxes for suspicious occlusion, item stacking, or tray crowding.
    """
    flagged = []
    total_items = len(detected_items)
    
    if total_items == 0:
        return AntiTheftReport(
            status="SECURE",
            occlusion_detected=False,
            message="Tray is empty. Waiting for items...",
            flagged_pairs=[],
            tray_density_score=0.0
        )

    # Check pairwise IoU
    for i in range(total_items):
        for j in range(i + 1, total_items):
            box1 = detected_items[i]["bbox"]
            box2 = detected_items[j]["bbox"]
            iou = compute_iou(box1, box2)

            if iou > threshold:
                name1 = detected_items[i]["name"]
                name2 = detected_items[j]["name"]
                flagged.append(f"{name1} overlaps with {name2} (IoU: {iou:.2f})")

    # Calculate tray coverage density
    img_h, img_w = image_shape[:2]
    total_tray_area = max(1, img_h * img_w)
    items_area = sum(item["bbox"][2] * item["bbox"][3] for item in detected_items)
    density = min(1.0, items_area / total_tray_area)

    if flagged:
        return AntiTheftReport(
            status="WARNING",
            occlusion_detected=True,
            message=f"⚠️ {len(flagged)} item overlap(s) detected. Please separate overlapping items on tray for accurate billing.",
            flagged_pairs=flagged,
            tray_density_score=round(density, 2)
        )

    return AntiTheftReport(
        status="SECURE",
        occlusion_detected=False,
        message="🟢 All items cleanly separated. Loss prevention verification passed.",
        flagged_pairs=[],
        tray_density_score=round(density, 2)
    )
