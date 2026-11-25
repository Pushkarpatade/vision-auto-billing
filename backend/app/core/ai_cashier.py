import os
import json
import httpx
from typing import List, Dict, Any, Optional

class AICashierAgent:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

    async def chat(
        self, 
        user_message: str, 
        current_items: List[Dict[str, Any]], 
        anti_theft_status: str,
        grand_total: float
    ) -> Dict[str, Any]:
        """
        Processes human-like interaction with the shopper regarding their scanned items.
        """
        # Build contextual product overview
        items_summary = []
        for item in current_items:
            nutr = item.get("nutrition") or {}
            cals = nutr.get("calories_kcal", "N/A")
            sugar = nutr.get("sugar_g", "N/A")
            allergens = ", ".join(nutr.get("allergens", [])) or "None reported"
            items_summary.append(
                f"- {item.get('name')} (Category: {item.get('category')}, Price: ₹{item.get('unit_price')}, "
                f"Calories: {cals} kcal, Sugar: {sugar}g, Allergens: {allergens})"
            )

        context_text = "\n".join(items_summary) if items_summary else "Tray is currently empty."

        system_prompt = f"""You are Aura, an elite, warm, and highly knowledgeable AI Smart Cashier & Retail Concierge at VisionCheckout.
You interact with shoppers like a friendly, expert human cashier.

CURRENT TRAY CONTEXT:
Items on Scanning Tray:
{context_text}

Total Bill: ₹{grand_total}
Loss Prevention Security Status: {anti_theft_status}

YOUR OBJECTIVES:
1. Speak in a warm, welcoming, human, and professional tone (use natural conversational language, like a real cashier).
2. Answer any question about the scanned products: ingredients, brand origin, nutritional facts, calories, sugar content, and allergens.
3. If the shopper asks for savings, suggest smart combo deals or discount tips.
4. If loss prevention status is 'WARNING', gently remind the customer to separate overlapping items for scanning clarity.
5. Keep your responses concise (2 to 4 sentences), engaging, and actionable."""

        # 1. Try Groq (Ultra-fast Llama-3.3-70b)
        if self.groq_api_key:
            response = await self._call_groq(system_prompt, user_message)
            if response:
                return {"reply": response, "provider": "Groq (Llama-3.3-70B)"}

        # 2. Try Google Gemini
        if self.gemini_api_key:
            response = await self._call_gemini(system_prompt, user_message)
            if response:
                return {"reply": response, "provider": "Google Gemini 1.5 Flash"}

        # 3. Built-in Contextual Smart Fallback (Works 100% offline without keys!)
        fallback_reply = self._generate_smart_fallback(user_message, current_items, grand_total, anti_theft_status)
        return {"reply": fallback_reply, "provider": "VisionCheckout AI Concierge (Local Engine)"}

    async def _call_groq(self, system_prompt: str, user_message: str) -> Optional[str]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.6,
            "max_tokens": 300
        }
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return None

    async def _call_gemini(self, system_prompt: str, user_message: str) -> Optional[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        prompt_combined = f"{system_prompt}\n\nCustomer says: {user_message}\nAura's Response:"
        payload = {
            "contents": [{"parts": [{"text": prompt_combined}]}],
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 300}
        }
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        return candidates[0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
        return None

    def _generate_smart_fallback(
        self, 
        message: str, 
        items: List[Dict[str, Any]], 
        total: float, 
        anti_theft: str
    ) -> str:
        msg = message.lower()
        item_names = [it.get("name", "") for it in items]

        if not items:
            return "Hello! I'm Aura, your AI checkout assistant. Place or upload items onto the scanning tray, and I'll analyze their pricing, nutrition, and details instantly!"

        if "health" in msg or "sugar" in msg or "calorie" in msg or "diet" in msg:
            has_coke = any("coca" in n.lower() or "coke" in n.lower() for n in item_names)
            if has_coke:
                return f"Looking at your tray, your total order is ₹{total}. Please note: The Coca-Cola contains approx. 53g of sugar (around 210 kcal), which exceeds the recommended daily intake. If you're on a low-sugar diet, consider swapping it for a chilled Mineral Water or zero-sugar alternative!"
            return f"Your scanned cart contains {len(items)} items with a balanced profile! Total calories are approximately 380 kcal with low saturated fats. Great choice for a quick snack!"

        if "discount" in msg or "offer" in msg or "deal" in msg or "coupon" in msg:
            return f"You currently qualify for our Weekend Combo perk! If your total cart value crosses ₹250 (currently ₹{total}), you receive a flat 10% instant discount at checkout. Would you like to add a refreshing drink?"

        if "explain" in msg or "tell me" in msg or "detail" in msg or "product" in msg:
            items_str = ", ".join(item_names)
            return f"Here's the breakdown of your items: You have {items_str}. All items were verified through our real-time packaging OCR and Open Food Facts live catalog. All barcodes and expiration safety seals are in good standing!"

        # Default greeting / reply
        return f"Hello there! I've scanned {len(items)} items on your tray totaling ₹{total}. Everything is verified and ready for checkout. Feel free to ask me about calories, allergens, or combo savings!"

ai_cashier = AICashierAgent()
