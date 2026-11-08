async function refreshDocuments() {
  const res = await fetch('/api/documents'); const data = await res.json();
  const box = document.getElementById('documents');
  if (!box) return data;
  box.replaceChildren();
  const count = document.getElementById('doc-count');
  if (count) count.textContent = data.documents.length;
  const ready = document.getElementById('collection-status');
  if (!data.documents.length) {
    const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'No reference materials yet'; box.append(empty);
    if (ready) ready.textContent = 'No documents indexed';
  } else if (!data.passages) {
    const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = 'Reference materials found, but their text could not be read.'; box.append(empty);
    if (ready) ready.textContent = 'Documents found, text not indexed';
  } else if (ready) {
    ready.textContent = `${data.passages} searchable passages`;
  }
  data.documents.forEach(name => { const item = document.createElement('div'); item.className = 'doc'; item.textContent = name.replace(/\.pdf$/i, ''); box.append(item); });
}
const progressPanel = document.getElementById('upload-progress');
const progressBar = document.getElementById('progress-bar');
const uploadStatus = document.getElementById('upload-status');
const progressDetail = document.getElementById('progress-detail');
function setIndexingBusy(busy) {
  const fileInput = document.getElementById('file-input');
  const reload = document.getElementById('reload');
  if (fileInput) fileInput.disabled = busy;
  if (reload) reload.disabled = busy;
  if (progressPanel) progressPanel.setAttribute('aria-busy', String(busy));
}
function startProgress(name) {
  progressPanel.hidden = false; progressPanel.classList.remove('error');
  document.getElementById('progress-file').textContent = name;
  uploadStatus.textContent = 'Preparing…'; progressDetail.textContent = ''; progressBar.removeAttribute('value');
  const warnings = document.getElementById('progress-warnings'); warnings.replaceChildren(); warnings.hidden = true;
}
function showProgress(event) {
  if (event.stage === 'heartbeat') return;
  if (event.file) document.getElementById('progress-file').textContent = event.file;
  if (event.stage === 'warning') {
    const warnings = document.getElementById('progress-warnings'); warnings.hidden = false;
    const item = document.createElement('li'); item.textContent = `${event.file || ''}${event.page ? ' · page ' + event.page : ''}: ${event.message}`; warnings.append(item); return;
  }
  const labels = { uploaded: 'Upload complete · preparing extraction', document: 'Reading document', extracting: 'Extracting text', ocr: 'Recognizing scanned text (OCR)', translating: 'Translating to English', indexing: 'Building searchable passages', page_complete: 'Page indexed', saving: 'Saving search index', complete: 'Ready to search', error: 'Processing failed' };
  uploadStatus.textContent = labels[event.stage] || event.stage;
  if (event.pages) { progressBar.max = event.pages; progressBar.value = event.stage === 'page_complete' ? event.page : event.page - 1; progressDetail.textContent = `Page ${event.page} of ${event.pages}${event.chunks ? ' · chunk ' + event.chunk + ' of ' + event.chunks : ''}`; }
  else if (event.stage === 'document') { progressBar.removeAttribute('value'); progressDetail.textContent = `Document ${event.document} of ${event.documents}`; }
  else if (event.stage === 'saving' || event.stage === 'uploaded') { progressBar.removeAttribute('value'); progressDetail.textContent = ''; }
  else if (event.stage === 'complete') { progressBar.max = 1; progressBar.value = 1; progressDetail.textContent = `${event.indexed_passages ?? event.passages} searchable passages${event.warnings ? ' · ' + event.warnings + ' warning(s); review below' : ''}`; }
  else if (event.stage === 'error') { progressPanel.classList.add('error'); progressDetail.textContent = event.message; progressBar.max = 1; progressBar.value = 0; }
}
function indexRequest(url, form = null) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); let offset = 0; let pending = ''; let terminal = null;
    xhr.open('POST', url); xhr.setRequestHeader('Accept', 'application/x-ndjson');
    if (form) {
      xhr.upload.onprogress = e => { if (e.lengthComputable) { progressBar.max = e.total; progressBar.value = e.loaded; uploadStatus.textContent = `Uploading · ${Math.round(e.loaded / e.total * 100)}%`; } };
      xhr.upload.onload = () => { uploadStatus.textContent = 'Upload sent · waiting for extraction'; progressBar.removeAttribute('value'); };
    }
    const consume = () => {
      pending += xhr.responseText.slice(offset); offset = xhr.responseText.length;
      let end; while ((end = pending.indexOf('\n')) >= 0) { const line = pending.slice(0, end); pending = pending.slice(end + 1); if (!line.trim()) continue; const event = JSON.parse(line); showProgress(event); if (['complete', 'error'].includes(event.stage)) terminal = event; }
    };
    xhr.onprogress = () => { if (xhr.status >= 200 && xhr.status < 300) { try { consume(); } catch { reject(new Error('Could not read processing progress. Refresh the collection to check the result.')); xhr.abort(); } } };
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) { let detail = 'Upload or refresh failed'; try { detail = JSON.parse(xhr.responseText).detail || detail; } catch { /* keep fallback */ } reject(new Error(detail)); return; }
      try { consume(); if (!terminal) throw new Error('Progress connection ended before completion. Refresh the collection to check the result.'); if (terminal.stage === 'error') throw new Error(terminal.message); resolve(terminal); } catch (error) { reject(error); }
    };
    xhr.onerror = () => reject(new Error('Connection lost. Server processing may continue; refresh the collection before retrying.'));
    xhr.send(form);
  });
}
function showIndexError(error) { progressPanel.classList.add('error'); uploadStatus.textContent = 'Could not finish'; progressDetail.textContent = error.message; progressBar.max = 1; progressBar.value = 0; }
async function uploadFiles(files) {
  if (!files.length) return;
  setIndexingBusy(true); let failed = 0; let warnings = 0;
  try {
    for (const [i, file] of files.entries()) {
      startProgress(`${file.name} (${i + 1}/${files.length})`);
      try {
        if (!file.name.toLowerCase().endsWith('.pdf')) throw new Error('Upload a PDF file.');
        if (file.size > 25 * 1024 * 1024) throw new Error('PDF must be 25 MB or smaller.');
        const form = new FormData(); form.append('file', file);
        const result = await indexRequest('/api/documents?progress=true', form);
        warnings += result.warnings || 0;
        await refreshDocuments();
      } catch (error) { failed++; showIndexError(error); }
    }
    if (files.length > 1) { uploadStatus.textContent = `${files.length - failed} of ${files.length} documents indexed${failed ? ' · ' + failed + ' failed' : ''}${warnings ? ' · ' + warnings + ' warnings' : ''}`; if (failed) progressPanel.classList.add('error'); }
  } finally { setIndexingBusy(false); }
}
const reloadButton = document.getElementById('reload');
if (reloadButton && progressPanel) {
  reloadButton.addEventListener('click', async () => {
    setIndexingBusy(true); startProgress('Refreshing reference collection');
    try { await indexRequest('/api/reload?progress=true'); await refreshDocuments(); } catch (error) { showIndexError(error); } finally { setIndexingBusy(false); }
  });
}
const fileInput = document.getElementById('file-input');
if (fileInput) {
  fileInput.addEventListener('change', async event => {
    const files = [...event.target.files];
    try { await uploadFiles(files); } finally { event.target.value = ''; }
  });
}
const dropzone = document.getElementById('dropzone');
if (dropzone) {
  dropzone.addEventListener('dragover', event => { event.preventDefault(); dropzone.classList.add('drag'); });
  dropzone.addEventListener('dragleave', event => { if (!dropzone.contains(event.relatedTarget)) dropzone.classList.remove('drag'); });
  dropzone.addEventListener('drop', async event => {
    event.preventDefault(); dropzone.classList.remove('drag');
    await uploadFiles([...event.dataTransfer.files]);
  });
}
refreshDocuments().catch(() => {});
