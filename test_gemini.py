import os
import httpx
import json
import base64
import cv2

img_path = r'C:\Users\Shree\.gemini\antigravity\brain\f1dc1b8c-a5cf-4836-823b-1467754a5b20\.user_uploaded\media_1795573211016_25fbba8b.jpg'
img = cv2.imread(img_path)
h, w = img.shape[:2]
scale = 640.0 / max(h, w)
resized = cv2.resize(img, (int(w * scale), int(h * scale)))
_, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
b64 = base64.b64encode(buf).decode('utf-8')

prompt = """Analyze this retail checkout tray image. Locate and identify all distinct consumer retail packaged products (bottles, snack packets, biscuit pouches, cans).
Return JSON array only:
[
  {
    "name": "Exact product name (e.g. Sprite Lime Drink, Aqua Velly Mineral Water, Oreo Biscuit, JimJam)",
    "category": "Beverage",
    "unit_price": 20.0,
    "box_2d": [ymin, xmin, ymax, xmax],
    "cognitive_reasoning": "Clear bottle with pink Aqua Velly label"
  }
]
Important rules:
- Bounding box [ymin, xmin, ymax, xmax] must cover the ENTIRE object (normalized 0 to 1000).
- NEVER detect caps, logos, or labels as separate items.
- Be precise in item count: each physical bottle or packet is exactly 1 item.
"""

key = 'AQ.Ab8RN6L2iPCM61gcOPVFtfRl_tpArOVeADNECEjY6goOADvrlA'
url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={key}'
payload = {
    'contents': [{
        'parts': [
            {'text': prompt},
            {'inline_data': {'mime_type': 'image/jpeg', 'data': b64}}
        ]
    }],
    'generationConfig': {
        'response_mime_type': 'application/json',
        'temperature': 0.1
    }
}

try:
    r = httpx.post(url, json=payload, timeout=20.0)
    print('Status:', r.status_code)
    if r.status_code == 200:
        res = r.json()
        text = res['candidates'][0]['content']['parts'][0]['text']
        items = json.loads(text)
        print(f'Detected {len(items)} items:')
        for it in items:
            print(f"  - {it['name']} (₹{it['unit_price']}) box={it.get('box_2d')}")
    else:
        print('Error:', r.text[:300])
except Exception as e:
    print('Exception:', e)
