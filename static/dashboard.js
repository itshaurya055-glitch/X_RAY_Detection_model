// Page navigation
function showPage(page) {
  document.querySelectorAll('.state').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelectorAll('.nav-item')[['analysis', 'history', 'settings'].indexOf(page)].classList.add('active');
  const titles = { analysis: 'TB Detection Portal', history: 'Analysis History', settings: 'Settings' };
  document.getElementById('page-title').textContent = titles[page];
}

// Drag and drop handling
const zone = document.getElementById('upload-zone');
zone.addEventListener('dragover', e => {
  e.preventDefault();
  zone.classList.add('drag-over');
});
zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});

// File handling
function handleFile(input) {
  if (input.files[0]) processFile(input.files[0]);
}

const API_URL = 'https://x-ray-detection-model.onrender.com';
let history_list = JSON.parse(localStorage.getItem('tb_history') || '[]');
let currentData = null;
let currentImgSrc = null;

function processFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    currentImgSrc = e.target.result;
    document.getElementById('state-upload').style.display = 'none';
    document.getElementById('state-processing').style.display = 'block';
    document.getElementById('processing-img').src = currentImgSrc;
    document.getElementById('upload-filename').textContent = file.name;
    document.getElementById('upload-badge').style.display = 'flex';
    callAPI(file, currentImgSrc);
  };
  reader.readAsDataURL(file);
}

async function callAPI(file, imgSrc) {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();

    if (!data.success) throw new Error(data.error || 'Prediction failed');

    showResult(data, imgSrc);

  } catch (err) {
    document.getElementById('state-processing').style.display = 'none';
    document.getElementById('state-upload').style.display = 'block';
    alert(`Error: ${err.message}`);
  }
}

function showResult(data, imgSrc) {
  currentData = data;
  document.getElementById('state-processing').style.display = 'none';
  document.getElementById('state-result').style.display = 'block';
  document.getElementById('result-xray').src = imgSrc;
  document.getElementById('result-filename').textContent = data.filename;

  const titleEl = document.getElementById('result-title');
  const barEl = document.getElementById('confidence-bar');
  const findEl = document.getElementById('finding-text');
  const bboxEl = document.getElementById('result-bbox');

  if (data.is_tb) {
    titleEl.textContent = data.label;
    titleEl.className = 'result-title danger';
    barEl.className = 'confidence-bar danger';
    findEl.className = 'finding-text';
    if (data.bbox) {
      bboxEl.style.cssText = `display:block;left:${data.bbox.left}%;top:${data.bbox.top}%;width:${data.bbox.width}%;height:${data.bbox.height}%`;
    }
  } else {
    titleEl.textContent = data.label;
    titleEl.className = 'result-title success';
    barEl.className = 'confidence-bar success';
    findEl.className = 'finding-text normal';
    bboxEl.style.display = 'none';
  }

  findEl.textContent = data.finding;
  document.getElementById('confidence-value').textContent = data.confidence + '%';
  document.getElementById('prob-normal').textContent = data.prob_normal + '%';
  document.getElementById('prob-tb').textContent = data.prob_tb + '%';

  if (data.gradcam_b64) {
    const camImg = document.getElementById('gradcam-img');
    const camPlh = document.getElementById('gradcam-placeholder');
    camImg.src = data.gradcam_b64;
    camImg.style.display = 'block';
    camPlh.style.display = 'none';
  }

  setTimeout(() => {
    barEl.style.width = data.confidence + '%';
  }, 100);

  autoSave(data, imgSrc);
}

function autoSave(data, imgSrc) {
  const entry = {
    id: Date.now(),
    filename: data.filename,
    is_tb: data.is_tb,
    confidence: data.confidence,
    prob_tb: data.prob_tb,
    prob_normal: data.prob_normal,
    finding: data.finding,
    gradcam: data.gradcam_b64,
    xray: imgSrc,
    date: new Date().toLocaleString('en-IN', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  };
  history_list.unshift(entry);
  localStorage.setItem('tb_history', JSON.stringify(history_list));
  renderHistory();
}

function renderHistory() {
  const grid = document.getElementById('history-grid');
  if (!history_list.length) {
    grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px;color:var(--text-muted);font-size:14px">No analyses yet. Upload an X-ray to get started.</div>';
    return;
  }
  grid.innerHTML = history_list
    .map(
      e => `
      <div class="history-card" onclick="viewHistory(${e.id})">
        <img src="${e.xray}" alt="xray"
             style="width:100%;height:160px;object-fit:cover;background:#0d1117;display:block;">
        <div class="history-info">
          <div class="history-name">${e.filename}</div>
          <div class="history-date">${e.date}</div>
          <span class="tag ${e.is_tb ? 'tb' : 'normal'}">
            ● ${e.is_tb ? 'Tuberculosis' : 'Normal'} - ${e.confidence}%
          </span>
        </div>
      </div>
    `
    )
    .join('');
}

function viewHistory(id) {
  const entry = history_list.find(e => e.id === id);
  if (!entry) return;
  showPage('analysis');
  showResult(entry, entry.xray);
}

function clearHistory() {
  if (!confirm('Clear all history?')) return;
  history_list = [];
  localStorage.removeItem('tb_history');
  renderHistory();
}

function downloadReport() {
  if (!currentData) {
    alert('No analysis to download yet!');
    return;
  }
  const d = currentData;
  const reportColor = d.is_tb ? '#c0392b' : '#1a7a4a';
  const findingBg = d.is_tb ? '#fdf0ef' : '#edf7f2';
  const html = [
    '<!DOCTYPE html>',
    '<html><head>',
    '<title>TB Detection Report</title>',
    '<style>',
    'body{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;padding:20px;color:#1a2333}',
    'h1{color:#1a6fb5;border-bottom:2px solid #1a6fb5;padding-bottom:10px}',
    `.result{font-size:24px;font-weight:700;color:${reportColor};margin:20px 0}`,
    'table{width:100%;border-collapse:collapse;margin:20px 0}',
    'td,th{padding:12px;border:1px solid #e2e8f4;font-size:14px}',
    'th{background:#f4f7fb;font-weight:600}',
    `.finding{background:${findingBg};border-left:4px solid ${reportColor};padding:16px;border-radius:8px;margin:20px 0}`,
    'img{max-width:100%;border-radius:8px;margin:10px 0}',
    '.footer{color:#999;font-size:12px;margin-top:40px;border-top:1px solid #eee;padding-top:16px}',
    '</style></head><body>',
    '<h1>TB Detection Report</h1>',
    `<p><strong>File:</strong> ${d.filename}</p>`,
    `<p><strong>Date:</strong> ${new Date().toLocaleString()}</p>`,
    `<div class="result">${d.label}</div>`,
    '<table>',
    '<tr><th>Metric</th><th>Value</th></tr>',
    `<tr><td>Confidence Score</td><td><strong>${d.confidence}%</strong></td></tr>`,
    `<tr><td>Normal Probability</td><td>${d.prob_normal}%</td></tr>`,
    `<tr><td>TB Probability</td><td>${d.prob_tb}%</td></tr>`,
    '<tr><td>Detection Threshold</td><td>35%</td></tr>',
    '<tr><td>Model</td><td>CheXNet DenseNet-121</td></tr>',
    '</table>',
    '<h3>Clinical Finding</h3>',
    `<div class="finding">${d.finding}</div>`,
    '<h3>Grad-CAM Heatmap</h3>',
    `<img src="${d.gradcam_b64}" alt="Grad-CAM">`,
    '<h3>Model Performance</h3>',
    '<table>',
    '<tr><th>Metric</th><th>Score</th></tr>',
    '<tr><td>Accuracy</td><td>88.24%</td></tr>',
    '<tr><td>AUC-ROC</td><td>93.36%</td></tr>',
    '<tr><td>Precision</td><td>97.14%</td></tr>',
    '<tr><td>Sensitivity</td><td>79.07%</td></tr>',
    '<tr><td>F1 Score</td><td>87.18%</td></tr>',
    '</table>',
    '<div class="footer">This report is generated by an AI model for research purposes only. Always consult a qualified medical professional for diagnosis. | TB Detect AI</div>',
    '</body></html>'
  ].join('');

  const blob = new Blob([html], { type: 'text/html' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `TB_Report_${d.filename.replace(/\.[^.]+$/, '')}_${Date.now()}.html`;
  a.click();
}

function resetToUpload() {
  document.getElementById('state-result').style.display = 'none';
  document.getElementById('state-processing').style.display = 'none';
  document.getElementById('state-upload').style.display = 'block';
  document.getElementById('file-input').value = '';
  document.getElementById('gradcam-img').style.display = 'none';
  document.getElementById('gradcam-placeholder').style.display = 'flex';
}

renderHistory();