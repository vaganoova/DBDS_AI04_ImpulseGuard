// ImpulseGuard popup logic — talks to the local FastAPI backend.

const API_BASE = "http://127.0.0.1:8000";

// Colors per impulse level (match the backend's level numbers).
const LEVEL_STYLE = {
  0: { bg: "#dcfce7", fg: "#166534", icon: "✅" },
  1: { bg: "#fef9c3", fg: "#854d0e", icon: "🟡" },
  2: { bg: "#ffedd5", fg: "#9a3412", icon: "⚠️" },
  3: { bg: "#fee2e2", fg: "#991b1b", icon: "🚨" },
};

let lastPurchase = null;  // remembered so feedback can be sent

// Populate the 1-10 frequency dropdown.
const freqSelect = document.getElementById("frequency");
for (let i = 1; i <= 10; i++) {
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = i;
  freqSelect.appendChild(opt);
}
freqSelect.value = 5;

function setStatus(msg) {
  document.getElementById("status").textContent = msg || "";
}

// Build the purchase object from the form (hour is automatic).
function readForm() {
  const price = parseFloat(document.getElementById("price").value);
  if (isNaN(price) || price < 0) {
    setStatus("Please enter a valid price.");
    return null;
  }
  return {
    hour: new Date().getHours(),
    price: price,
    category: document.getElementById("category").value,
    frequency: parseInt(document.getElementById("frequency").value, 10),
    is_essential: parseInt(document.getElementById("is_essential").value, 10),
    deliberation_minutes: parseInt(document.getElementById("deliberation").value, 10),
    on_wishlist: parseInt(document.getElementById("on_wishlist").value, 10),
  };
}

function showResult(data) {
  const style = LEVEL_STYLE[data.level];
  const result = document.getElementById("result");
  result.style.background = style.bg;
  result.style.color = style.fg;
  result.querySelector(".title").textContent = `${style.icon} ${data.level_name}`;
  result.querySelector(".msg").textContent =
    data.level === 0 ? "This looks planned and low-risk."
    : data.level === 3 ? "Emotional, unplanned pattern — wait 24 hours before buying."
    : "Worth a second thought before you buy.";
  result.querySelector(".conf").textContent = `Confidence: ${Math.round(data.confidence * 100)}%`;
  result.style.display = "block";
  document.getElementById("feedback").style.display = "block";
}

// --- Check button -------------------------------------------------------
document.getElementById("check").addEventListener("click", async () => {
  setStatus("");
  const purchase = readForm();
  if (!purchase) return;
  lastPurchase = purchase;
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(purchase),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    showResult(await res.json());
  } catch (e) {
    setStatus("Could not reach ImpulseGuard backend. Is the server running on :8000?");
  }
});

// --- Feedback buttons ---------------------------------------------------
document.querySelectorAll(".fb-grid button").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (!lastPurchase) return;
    const trueLevel = parseInt(btn.dataset.level, 10);
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...lastPurchase, true_level: trueLevel }),
      });
      setStatus("Thanks! Feedback saved for the next training run.");
      document.getElementById("feedback").style.display = "none";
    } catch (e) {
      setStatus("Could not save feedback (backend offline?).");
    }
  });
});

// --- Best-effort: detect a price on the current page --------------------
document.getElementById("detectPrice").addEventListener("click", async () => {
  setStatus("");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const m = document.body.innerText.match(/(?:€|\$|EUR|USD)\s?(\d{1,5}(?:[.,]\d{2})?)/);
        return m ? m[1].replace(",", ".") : null;
      },
    });
    if (result) {
      document.getElementById("price").value = result;
    } else {
      setStatus("No price found on this page — enter it manually.");
    }
  } catch (e) {
    setStatus("Could not read the page — enter the price manually.");
  }
});
