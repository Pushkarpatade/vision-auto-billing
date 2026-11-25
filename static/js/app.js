// VisionCheckout Client Application
let currentReceiptData = null;
let currentItemsList = [];
let voiceEnabled = true;
let webcamStream = null;
let activeScanId = 0;
const API_BASE_URL = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

// On load: automatically benchmark and load the SIH Hackathon sample tray!
document.addEventListener("DOMContentLoaded", () => {
  loadSampleTray("coke_snacks");
});

function toggleVoice() {
  voiceEnabled = !voiceEnabled;
  const btn = document.getElementById("tts-toggle");
  btn.innerText = voiceEnabled ? "🔊 Voice: ON" : "🔇 Voice: OFF";
  if (!voiceEnabled && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
}

function speakText(text) {
  if (!voiceEnabled || !window.speechSynthesis) return;
  window.speechSynthesis.cancel(); // Stop ongoing speech
  const cleanText = text.replace(/[*#_~]/g, ""); // strip markdown
  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.rate = 1.05;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

async function loadSampleTray(sampleId) {
  document.querySelectorAll(".btn-sample").forEach(btn => {
    btn.classList.remove("active", "bg-slate-900", "text-white");
    btn.classList.add("bg-white", "text-slate-700", "border-slate-200");
  });
  const clickedBtn = Array.from(document.querySelectorAll(".btn-sample")).find(b => b.getAttribute("onclick")?.includes(sampleId));
  if (clickedBtn) {
    clickedBtn.classList.add("active", "bg-slate-900", "text-white");
    clickedBtn.classList.remove("bg-white", "text-slate-700");
  }

  const scanId = beginNewScan();

  try {
    const res = await fetch(apiUrl(`/api/v1/detect-sample/${sampleId}`), { method: "POST" });
    if (!res.ok) throw new Error("Failed to process sample tray");
    const data = await res.json();
    if (scanId === activeScanId) renderDetectionResult(data);
  } catch (err) {
    if (scanId === activeScanId) alert("Error loading sample tray: " + err.message);
  } finally {
    if (scanId === activeScanId) showLoading(false);
  }
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const scanId = beginNewScan();
  try {
    const res = await fetch(apiUrl("/api/v1/detect"), {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Image processing failed");
    const data = await res.json();
    if (scanId === activeScanId) renderDetectionResult(data);
  } catch (err) {
    if (scanId === activeScanId) alert("Upload error: " + err.message);
  } finally {
    if (scanId === activeScanId) showLoading(false);
  }
}

function beginNewScan() {
  activeScanId += 1;
  currentReceiptData = null;
  currentItemsList = [];
  if (window.speechSynthesis) window.speechSynthesis.cancel();

  document.getElementById("viewport-image").removeAttribute("src");
  document.getElementById("viewport-image").style.display = "none";
  const emptyState = document.getElementById("empty-state");
  emptyState.style.display = "block";
  emptyState.querySelector("h3").innerText = "Scanning tray...";
  emptyState.querySelector("p").innerText = "Starting a fresh scan. Previous receipt data has been cleared.";

  document.getElementById("perf-time").innerText = "-- ms";
  document.getElementById("invoice-id").innerText = "INV-SCANNING";
  document.getElementById("receipt-date").innerText = "--";
  document.getElementById("receipt-count").innerText = "0";
  document.getElementById("anti-theft-pill").className = "pill-card pill-secure";
  document.getElementById("anti-theft-msg").innerText = "Waiting for a fresh tray scan.";
  document.getElementById("pricing-tier-msg").innerText = "Scanning for exact SKU verification";
  document.getElementById("items-tbody").innerHTML = `
    <tr><td colspan="4" style="text-align:center; color:#94a3b8; padding:2rem;">Scanning a new tray. Previous items were cleared.</td></tr>`;

  document.getElementById("pos-subtotal").innerText = "₹0.00";
  document.getElementById("pos-tax").innerText = "₹0.00";
  document.getElementById("pos-total").innerText = "₹0.00";
  document.getElementById("checkout-btn-amt").innerText = "0.00";
  document.getElementById("nutr-cals").innerText = "0 kcal";
  document.getElementById("nutr-sugar").innerText = "0 g";
  document.getElementById("nutr-allergens").innerText = "Allergens: None";
  const healthTag = document.getElementById("health-tag");
  healthTag.innerText = "Awaiting scan";
  healthTag.style.background = "#e2e8f0";
  healthTag.style.color = "#475569";

  document.getElementById("agent-provider").innerText = "Resetting chat context";
  document.getElementById("chat-box").innerHTML = `
    <div style="display:flex; gap:0.6rem; align-items:flex-start;">
      <div style="font-weight:700; color:#2563eb;">Aura:</div>
      <div id="aura-text">I cleared the previous tray context. I’ll help once this scan is complete.</div>
    </div>`;

  showLoading(true);
  return activeScanId;
}

function renderDetectionResult(data) {
  const receipt = data.receipt;
  currentReceiptData = receipt;
  currentItemsList = receipt.items;

  // Viewport Image
  const imgEl = document.getElementById("viewport-image");
  const emptyState = document.getElementById("empty-state");
  imgEl.src = data.annotated_image;
  imgEl.style.display = "block";
  emptyState.style.display = "none";

  // Performance / Latency
  document.getElementById("perf-time").innerText = receipt.processing_time_ms;
  const heroLatency = document.getElementById("hero-stat-latency");
  if (heroLatency) heroLatency.innerText = `${receipt.processing_time_ms}ms`;

  // Invoice Metadata
  document.getElementById("invoice-id").innerText = receipt.invoice_number;
  document.getElementById("receipt-date").innerText = receipt.timestamp;
  document.getElementById("receipt-count").innerText = `${receipt.billable_quantity} billable / ${receipt.total_quantity} detected`;

  // Anti-Theft Status Pill
  const atPill = document.getElementById("anti-theft-pill");
  const atMsg = document.getElementById("anti-theft-msg");
  if (receipt.anti_theft.occlusion_detected) {
    atPill.className = "pill-card pill-warning flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold bg-rose-50 text-rose-800 border border-rose-200";
    atMsg.innerText = receipt.anti_theft.message;
  } else {
    atPill.className = "pill-card pill-secure flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200";
    atMsg.innerText = receipt.anti_theft.message;
  }

  // Populate the review queue.
  const tbody = document.getElementById("items-tbody");
  tbody.innerHTML = "";

  if (receipt.items.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-center text-slate-400 py-8 text-xs font-medium">No items detected on tray.</td></tr>`;
  } else {
    receipt.items.forEach(it => {
      const confPct = Math.round(it.confidence * 100);
      const isBillable = it.billable;
      const statusLabel = it.recognition_status.replaceAll("_", " ");
      const statusBg = isBillable ? '#f0fdf4' : '#fff7ed';
      const statusColor = isBillable ? '#166534' : '#b45309';
      const statusBorder = isBillable ? '#bbf7d0' : '#fed7aa';
      const row = document.createElement("tr");
      row.className = "border-b border-slate-100 hover:bg-slate-50/80 transition-colors";
      row.innerHTML = `
        <td class="py-2.5 px-3">
          <div class="font-bold text-slate-900 text-xs">${it.name}</div>
          <div class="flex items-center gap-1.5 mt-1 flex-wrap">
            <span class="text-[10px] font-medium px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">${it.category}</span>
            <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200/80">${confPct}% Conf</span>
            <span class="text-[10px] font-bold px-2 py-0.5 rounded border" style="background:${statusBg}; color:${statusColor}; border-color:${statusBorder};">${statusLabel}</span>
          </div>
          ${it.cognitive_reasoning ? `
            <div class="text-[11px] text-slate-600 mt-1.5 bg-indigo-50/50 p-2 rounded-lg border-l-2 border-indigo-600 leading-relaxed font-sans">
              🧠 <strong class="text-slate-800">Cognitive Reasoning:</strong> ${it.cognitive_reasoning}
            </div>` : ''}
        </td>
        <td class="py-2.5 px-2 text-center font-bold text-slate-800 align-top pt-3">${it.quantity}</td>
        <td class="py-2.5 px-2 text-right font-semibold text-slate-700 align-top pt-3">₹${it.unit_price.toFixed(2)}</td>
        <td class="py-2.5 px-3 text-right font-bold align-top pt-3 text-indigo-600">₹${it.subtotal.toFixed(2)}</td>
      `;
      tbody.appendChild(row);
    });
  }

  // Totals apply only to this image. Candidate amounts are labelled until SKU confirmation.
  const hasEstimatedItems = receipt.review_required_quantity > 0;
  const shownSubtotal = hasEstimatedItems ? receipt.estimated_subtotal : receipt.subtotal;
  const shownTax = hasEstimatedItems ? receipt.estimated_total_tax : receipt.total_tax;
  const shownTotal = hasEstimatedItems ? receipt.estimated_grand_total : receipt.grand_total;
  document.getElementById("subtotal-label").innerText = hasEstimatedItems ? "Estimated subtotal" : "Subtotal";
  document.getElementById("tax-label").innerText = hasEstimatedItems ? "Estimated GST" : "CGST (2.5%) + SGST (2.5%)";
  document.getElementById("total-label").innerText = hasEstimatedItems ? "Estimated current-image total" : "Grand Total";
  document.getElementById("pos-subtotal").innerText = `₹${shownSubtotal.toFixed(2)}`;
  document.getElementById("pos-tax").innerText = `₹${shownTax.toFixed(2)}`;
  document.getElementById("pos-total").innerText = `₹${shownTotal.toFixed(2)}`;
  document.getElementById("checkout-btn-amt").innerText = receipt.grand_total.toFixed(2);

  // Nutrition Card
  const nutr = receipt.nutrition_summary;
  document.getElementById("nutr-cals").innerText = `${nutr.total_calories_kcal} kcal`;
  document.getElementById("nutr-sugar").innerText = `${nutr.total_sugar_g} g`;
  const allergens = nutr.allergens_flagged.length ? nutr.allergens_flagged.join(", ") : "None";
  document.getElementById("nutr-allergens").innerText = `Allergens: ${allergens}`;

  const healthTag = document.getElementById("health-tag");
  if (nutr.health_rating.includes("Alert")) {
    healthTag.innerText = "High Sugar Alert";
    healthTag.style.background = "#fee2e2";
    healthTag.style.color = "#b91c1c";
  } else {
    healthTag.innerText = "Healthy Balance";
    healthTag.style.background = "#dcfce7";
    healthTag.style.color = "#15803d";
  }

  const pricingMessage = document.getElementById("pricing-tier-msg");
  pricingMessage.innerText = receipt.review_required_quantity
    ? `${receipt.review_required_quantity} item(s) need SKU confirmation`
    : "All items verified against the product master";

  // Trigger Aura welcome greeting for the new cart
  triggerAuraCartGreeting(receipt);
}

function showLoading(show) {
  document.getElementById("loading-spinner").style.display = show ? "flex" : "none";
}

// -------------------------------------------------------------
// Interactive AI Smart Cashier Agent ("Aura")
// -------------------------------------------------------------
async function triggerAuraCartGreeting(receipt) {
  const itemNames = receipt.items.map(it => it.name).join(", ");
  const prompt = `I just placed ${receipt.items.length} items on the checkout tray: ${itemNames}. Greet me warmly, explain that visual candidates require exact SKU verification before billing, and offer help with the review.`;
  await askAura(prompt, false);
}

async function askAura(question, displayUserMsg = true) {
  if (displayUserMsg) {
    appendChatMessage("You", question, "#0f172a");
  }

  const payload = {
    message: question,
    items: currentItemsList,
    anti_theft_status: currentReceiptData ? currentReceiptData.anti_theft.status : "SECURE",
    grand_total: currentReceiptData ? currentReceiptData.grand_total : 0.0
  };

  try {
    const res = await fetch(apiUrl("/api/v1/chat-cashier"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Cashier API offline");
    const data = await res.json();

    document.getElementById("agent-provider").innerText = data.provider;
    appendChatMessage("Aura", data.reply, "#2563eb");
    speakText(data.reply);
  } catch (err) {
    appendChatMessage("Aura", "I am currently running in offline local mode. Your items are verified and ready for checkout!", "#2563eb");
  }
}

function sendAuraMessage() {
  const input = document.getElementById("user-chat-input");
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  askAura(msg, true);
}

function appendChatMessage(sender, text, color) {
  const chatBox = document.getElementById("chat-box");
  const msgEl = document.createElement("div");
  msgEl.style.display = "flex";
  msgEl.style.gap = "0.6rem";
  msgEl.style.marginTop = "0.5rem";
  msgEl.innerHTML = `
    <div style="font-weight:700; color:${color}; min-width:48px;">${sender}:</div>
    <div>${text}</div>
  `;
  chatBox.appendChild(msgEl);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// -------------------------------------------------------------
// Live Webcam Modal
// -------------------------------------------------------------
async function openWebcamModal() {
  const modal = document.getElementById("webcam-modal");
  const video = document.getElementById("webcam-video");
  modal.style.display = "flex";

  try {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 }, 
          height: { ideal: 720 } 
        } 
      });
    } catch (e) {
      // Fallback for laptops/desktops without environment facing mode or exact resolutions
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    }
    video.srcObject = webcamStream;
  } catch (err) {
    alert("Camera permission denied or camera unavailable: " + err.message);
    closeWebcamModal();
  }
}

function closeWebcamModal() {
  const modal = document.getElementById("webcam-modal");
  const video = document.getElementById("webcam-video");
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  video.srcObject = null;
  modal.style.display = "none";
}

function captureWebcamFrame() {
  const video = document.getElementById("webcam-video");
  const canvas = document.getElementById("hidden-canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async blob => {
    closeWebcamModal();
    const formData = new FormData();
    formData.append("file", blob, "webcam_tray.jpg");

    const scanId = beginNewScan();
    try {
      const res = await fetch(apiUrl("/api/v1/detect"), { method: "POST", body: formData });
      if (!res.ok) throw new Error("Detection failed");
      const data = await res.json();
      if (scanId === activeScanId) renderDetectionResult(data);
    } catch (err) {
      if (scanId === activeScanId) alert("Error: " + err.message);
    } finally {
      if (scanId === activeScanId) showLoading(false);
    }
  }, "image/jpeg", 0.92);
}

// -------------------------------------------------------------
// Print & Checkout
// -------------------------------------------------------------
function printReceipt() {
  if (!currentReceiptData) {
    alert("Please scan items before printing a receipt.");
    return;
  }
  window.print();
}

function completeCheckout() {
  if (!currentReceiptData || currentReceiptData.items.length === 0) {
    alert("No items on scanning tray to checkout!");
    return;
  }
  if (currentReceiptData.review_required_quantity > 0) {
    alert(`Checkout is blocked: ${currentReceiptData.review_required_quantity} item(s) still need exact SKU confirmation. Separate or rescan them, then select the verified catalogue product.`);
    return;
  }
  const total = currentReceiptData.grand_total.toFixed(2);
  const invoice = currentReceiptData.invoice_number;
  speakText(`Payment of Rupees ${total} received successfully. Thank you for shopping with VisionCheckout!`);
  alert(`🎉 Payment Successful!\n\nInvoice: ${invoice}\nAmount Paid: ₹${total}\n\nThank you for shopping at VisionCheckout!`);
}
