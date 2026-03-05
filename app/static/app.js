const form = document.getElementById('upload-form');
const fileInput = document.getElementById('video-file');
const statusEl = document.getElementById('status');
const table = document.getElementById('result-table');
const tbody = table.querySelector('tbody');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;

  const data = new FormData();
  data.append('file', fileInput.files[0]);

  statusEl.textContent = '解析中です...動画の長さに応じて時間がかかります。';
  table.hidden = true;
  tbody.innerHTML = '';

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: data });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const payload = await res.json();

    payload.corners.forEach((c) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${c.start_sec.toFixed(1)}</td>
        <td>${c.end_sec.toFixed(1)}</td>
        <td>${c.title}</td>
        <td>${c.summary}</td>
      `;
      tbody.appendChild(tr);
    });

    table.hidden = false;
    statusEl.textContent = `解析完了: ${payload.filename}`;
  } catch (err) {
    statusEl.textContent = `エラー: ${err.message}`;
  }
});
