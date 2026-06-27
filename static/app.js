const sampleFields = [
  ["total_trades", "Total trades", 20],
  ["avg_pnl", "Average PnL", 120.5],
  ["total_pnl", "Total PnL", 2410],
  ["win_rate", "Win rate", 0.55],
  ["avg_trade_size", "Average trade size", 725],
  ["total_trade_size", "Total trade size", 14500],
  ["buy_ratio", "Buy ratio", 0.6],
  ["sell_ratio", "Sell ratio", 0.4],
  ["pnl_per_trade", "PnL per trade", 120.5],
];

const fieldsContainer = document.getElementById("fields");
const resultContainer = document.getElementById("result");
const modelInfo = document.getElementById("modelInfo");
const refreshButton = document.getElementById("refreshModel");
const form = document.getElementById("predictForm");

function renderFields() {
  fieldsContainer.innerHTML = sampleFields
    .map(
      ([name, label, value]) => `
        <div class="field">
          <label for="${name}">${label}</label>
          <input id="${name}" name="${name}" type="number" step="any" value="${value}" />
        </div>
      `
    )
    .join("");
}

async function fetchModelInfo() {
  const response = await fetch("/api/model");
  const data = await response.json();
  modelInfo.textContent = JSON.stringify(data, null, 2);
}

function readFormValues() {
  const payload = {};
  for (const [name] of sampleFields) {
    payload[name] = Number(document.getElementById(name).value);
  }
  return payload;
}

function renderResult(data) {
  const className = String(data.prediction).toLowerCase().includes("fear") ? "bad" : "good";
  const probabilities = Object.entries(data.probabilities)
    .map(([label, value]) => `<div class="result-item"><span>${label}</span><strong>${(value * 100).toFixed(2)}%</strong></div>`)
    .join("");

  resultContainer.classList.remove("empty");
  resultContainer.innerHTML = `
    <div class="label ${className}">${data.prediction}</div>
    <div class="result-item"><span>Confidence</span><strong>${(data.confidence * 100).toFixed(2)}%</strong></div>
    <div class="result-list">${probabilities}</div>
  `;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultContainer.textContent = "Running prediction...";

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(readFormValues()),
  });

  const data = await response.json();
  if (!response.ok) {
    resultContainer.textContent = data.detail || "Prediction failed";
    return;
  }

  renderResult(data);
});

refreshButton.addEventListener("click", async () => {
  refreshButton.disabled = true;
  refreshButton.textContent = "Reloading...";
  try {
    const response = await fetch("/api/reload", { method: "POST" });
    const data = await response.json();
    modelInfo.textContent = JSON.stringify(data.metadata, null, 2);
  } finally {
    refreshButton.disabled = false;
    refreshButton.textContent = "Reload model";
    await fetchModelInfo();
  }
});

renderFields();
fetchModelInfo().catch(() => {
  modelInfo.textContent = "Model not trained yet. Run train_model.py first.";
});
