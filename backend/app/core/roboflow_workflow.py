"""Adapter for the optional hosted Roboflow product-recognition workflow.

Workflow output varies with its configured blocks. This adapter accepts common
object-detection output shapes, but only explicitly mapped labels can resolve
to a store SKU.
"""

import json
import os
import tempfile
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from anyio import to_thread

from backend.app.config import settings
from backend.app.schemas import NutritionData


class RoboflowWorkflowError(RuntimeError):
    """The hosted workflow is unavailable or incorrectly configured."""


def _normalized_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return confidence / 100.0 if confidence > 1 else confidence


def _prediction_lists(value: Any) -> Iterable[List[Dict[str, Any]]]:
    """Yield detection lists nested inside common workflow result containers."""
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            if any(any(key in item for key in ("class", "label", "class_name")) for item in value):
                yield value
        for item in value:
            yield from _prediction_lists(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"predictions", "detections"}:
                yield from _prediction_lists(item)
            elif isinstance(item, (dict, list)):
                yield from _prediction_lists(item)


@lru_cache(maxsize=1)
def _label_to_sku() -> Dict[str, str]:
    try:
        mapping = json.loads(settings.ROBOFLOW_LABEL_TO_SKU_JSON)
    except json.JSONDecodeError as exc:
        raise RoboflowWorkflowError("ROBOFLOW_LABEL_TO_SKU_JSON must be valid JSON.") from exc
    if not isinstance(mapping, dict) or not all(isinstance(label, str) and isinstance(sku, str) for label, sku in mapping.items()):
        raise RoboflowWorkflowError("ROBOFLOW_LABEL_TO_SKU_JSON must map label strings to SKU strings.")
    return {label.strip().lower(): sku.strip() for label, sku in mapping.items()}


def _prediction_bbox(prediction: Dict[str, Any], image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
    """Convert Roboflow centre or corner coordinates to x, y, width, height."""
    try:
        if all(key in prediction for key in ("x1", "y1", "x2", "y2")):
            left, top = float(prediction["x1"]), float(prediction["y1"])
            right, bottom = float(prediction["x2"]), float(prediction["y2"])
        else:
            width, height = float(prediction.get("width", prediction.get("w"))), float(prediction.get("height", prediction.get("h")))
            left, top = float(prediction["x"]) - width / 2, float(prediction["y"]) - height / 2
            right, bottom = left + width, top + height
    except (KeyError, TypeError, ValueError):
        return None
    left, top = max(0, int(round(left))), max(0, int(round(top)))
    right, bottom = min(image_width, int(round(right))), min(image_height, int(round(bottom)))
    return (left, top, right - left, bottom - top) if right > left and bottom > top else None


class RoboflowWorkflowDetector:
    def __init__(self) -> None:
        if not settings.ROBOFLOW_API_KEY:
            raise RoboflowWorkflowError("ROBOFLOW_API_KEY is required when ROBOFLOW_ENABLED=true.")
        self.label_to_sku = _label_to_sku()
        try:
            from inference_sdk import InferenceConfiguration, InferenceHTTPClient
        except ImportError as exc:
            raise RoboflowWorkflowError("inference-sdk is not installed. Run pip install -r requirements.txt.") from exc
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=settings.ROBOFLOW_API_KEY,
        ).configure(InferenceConfiguration(api_key_transport="header"))

    async def detect(self, image: np.ndarray, catalogue: Any) -> List[Dict[str, Any]]:
        """Call the blocking SDK off the FastAPI event loop and normalize results."""
        success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not success:
            raise RoboflowWorkflowError("Could not encode image for Roboflow inference.")
        source_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as source_file:
                source_file.write(encoded.tobytes())
                source_path = source_file.name
            result = await to_thread.run_sync(lambda: self.client.run_workflow(
                workspace_name=settings.ROBOFLOW_WORKSPACE,
                workflow_id=settings.ROBOFLOW_WORKFLOW_ID,
                images={"image": source_path},
                use_cache=True,
            ))
        except Exception as exc:
            raise RoboflowWorkflowError(f"Roboflow workflow request failed: {exc}") from exc
        finally:
            if source_path and os.path.exists(source_path):
                os.remove(source_path)
        return self._to_candidates(result, image.shape[1], image.shape[0], catalogue)

    def _to_candidates(self, result: Any, image_width: int, image_height: int, catalogue: Any) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()
        for prediction_list in _prediction_lists(result):
            for prediction in prediction_list:
                label = str(prediction.get("class") or prediction.get("label") or prediction.get("class_name") or "unknown").strip()
                confidence = _normalized_confidence(prediction.get("confidence", prediction.get("score", 0)))
                bbox = _prediction_bbox(prediction, image_width, image_height)
                key = (label.lower(), bbox)
                if not bbox or confidence < settings.ROBOFLOW_CONFIDENCE_THRESHOLD or key in seen:
                    continue
                seen.add(key)
                sku = self.label_to_sku.get(label.lower())
                product = catalogue.get_product(sku) if sku else None
                if product and product["active"]:
                    name, category, price = product["name"], product["category"], product["price"]
                    nutrition = NutritionData(**product["nutrition"]) if product["nutrition"] else None
                else:
                    sku, name, category, price, nutrition = None, label, "Unmapped", 0.0, None
                candidates.append({
                    "product_key": sku or label.lower().replace(" ", "_"),
                    "name": name, "category": category, "confidence": round(confidence, 4),
                    "unit_price": price, "quantity": 1, "subtotal": price, "bbox": bbox,
                    "detector_class": label, "_detection_score": confidence,
                    "pricing_source": "ROBOFLOW_STORE_CATALOG" if sku else "ROBOFLOW_UNMAPPED_LABEL",
                    "cognitive_reasoning": f"Roboflow workflow detected '{label}' at {confidence:.0%} confidence.",
                    "nutrition": nutrition, "roboflow_sku": sku, "roboflow_label": label,
                })
        return candidates
