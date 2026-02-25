const form = document.getElementById('upload-form');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const metricsEl = document.getElementById('metrics');
const tipsEl = document.getElementById('tips');

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const input = document.getElementById('video');
  if (!input.files.length) return;

  const data = new FormData();
  data.append('file', input.files[0]);

  statusEl.textContent = 'Analyzing... this can take 10-30 seconds on CPU.';
  resultsEl.hidden = true;
  metricsEl.innerHTML = '';
  tipsEl.innerHTML = '';

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: data,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Analysis failed');
    }

    const payload = await response.json();
    renderResults(payload);
    statusEl.textContent = 'Done.';
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
});

function renderResults(payload) {
  const { metrics, tips } = payload;
  resultsEl.hidden = false;

  metricsEl.innerHTML = Object.entries(metrics)
    .map(([key, value]) => `<div class="metric"><span>${formatKey(key)}</span><strong>${value}</strong></div>`)
    .join('');

  tipsEl.innerHTML = tips.map((tip) => `<li>${tip}</li>`).join('');
}

function formatKey(key) {
  return key.replaceAll('_', ' ');
}
