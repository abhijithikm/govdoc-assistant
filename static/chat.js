const messagesEl = document.getElementById('messages');
const history = [];
function appendInlineMarkdown(element, text) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g);
  for (const part of parts) {
    if (part.startsWith('**') && part.endsWith('**')) {
      const strong = document.createElement('strong'); strong.textContent = part.slice(2, -2); element.append(strong);
    } else if (part.startsWith('`') && part.endsWith('`')) {
      const code = document.createElement('code'); code.textContent = part.slice(1, -1); element.append(code);
    } else if (part.startsWith('*') && part.endsWith('*')) {
      const em = document.createElement('em'); em.textContent = part.slice(1, -1); element.append(em);
    } else if (/^\[[^\]]+\]\(https?:\/\//.test(part)) {
      const match = part.match(/^\[([^\]]+)\]\((.+)\)$/); const link = document.createElement('a');
      link.textContent = match[1]; link.href = match[2]; link.target = '_blank'; link.rel = 'noopener noreferrer'; element.append(link);
    } else {
      element.append(document.createTextNode(part));
    }
  }
}
function renderMessage(body, role, text) {
  if (role === 'user') { body.textContent = text; return; }
  const lines = text.split('\n'); let list = null; let paragraph = [];
  const flushList = () => { if (list) { body.append(list); list = null; } };
  const flushParagraph = () => { if (paragraph.length) { const p = document.createElement('p'); appendInlineMarkdown(p, paragraph.join(' ')); body.append(p); paragraph = []; } };
  const cells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('```')) {
      flushParagraph(); flushList(); const code = [];
      while (++i < lines.length && !lines[i].trim().startsWith('```')) code.push(lines[i]);
      const pre = document.createElement('pre'); const el = document.createElement('code'); el.textContent = code.join('\n'); pre.append(el); body.append(pre); continue;
    }
    if (line.includes('|') && i + 1 < lines.length && cells(lines[i + 1]).every(cell => /^:?-{3,}:?$/.test(cell))) {
      flushParagraph(); flushList(); const wrap = document.createElement('div'); wrap.className = 'table-wrap';
      const table = document.createElement('table'); const head = document.createElement('thead'); const row = document.createElement('tr');
      for (const cell of cells(line)) { const th = document.createElement('th'); th.scope = 'col'; appendInlineMarkdown(th, cell); row.append(th); } head.append(row); table.append(head);
      const tbody = document.createElement('tbody'); i++;
      while (i + 1 < lines.length && lines[i + 1].includes('|')) { const tr = document.createElement('tr'); for (const cell of cells(lines[++i])) { const td = document.createElement('td'); appendInlineMarkdown(td, cell); tr.append(td); } tbody.append(tr); }
      table.append(tbody); wrap.append(table); body.append(wrap); continue;
    }
    const heading = line.match(/^#{1,6}\s+(.+)/);
    if (heading) { flushParagraph(); flushList(); const h = document.createElement('h3'); appendInlineMarkdown(h, heading[1]); body.append(h); continue; }
    const bullet = line.match(/^(?:([-*•])|\d+[.)])\s+(.+)/);
    if (bullet) {
      flushParagraph(); const type = bullet[1] ? 'ul' : 'ol';
      if (list && list.tagName.toLowerCase() !== type) flushList();
      if (!list) { list = document.createElement(type); if (type === 'ol') list.start = parseInt(line, 10); }
      const item = document.createElement('li'); appendInlineMarkdown(item, bullet[2]); list.append(item); continue;
    }
    if (!line) { flushParagraph(); if (list && !/^\s*(?:[-*•]|\d+[.)])\s+/.test(lines[i + 1] || '')) flushList(); continue; }
    flushList(); paragraph.push(line);
  }
  flushParagraph(); flushList();
}
function addMessage(role, text, citations = []) {
  document.getElementById('welcome')?.remove();
  const row = document.createElement('div'); row.className = `message ${role}`;
  const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = role === 'user' ? 'You' : '✦';
  const body = document.createElement('div'); body.className = 'bubble'; renderMessage(body, role, text);
  if (citations.length) {
    const refs = document.createElement('div'); refs.className = 'citations'; const label = document.createElement('div'); label.textContent = 'Sources'; refs.append(label);
    const grouped = new Map(); for (const c of citations) { if (!grouped.has(c.source)) grouped.set(c.source, new Set()); grouped.get(c.source).add(c.page); }
    for (const [source, pages] of grouped) { const chip = document.createElement('span'); chip.className = 'source-chip'; chip.textContent = `${source} · pp. ${[...pages].sort((a, b) => a - b).join(', ')}`; refs.append(chip); } body.append(refs);
  }
  row.append(avatar, body); messagesEl.appendChild(row); messagesEl.scrollTop = messagesEl.scrollHeight; return row;
}
async function refreshStatus() {
  const res = await fetch('/api/documents'); const data = await res.json();
  document.querySelector('.online').textContent = data.passages ? '●  Documents ready' : '●  No documents indexed';
}
document.getElementById('chat-form').addEventListener('submit', async event => {
  event.preventDefault(); const input = document.getElementById('question'); const question = input.value.trim(); if (!question) return;
  addMessage('user', question); input.value = ''; input.style.height = 'auto';
  const send = document.getElementById('send'); send.disabled = true; send.textContent = '…';
  const pending = addMessage('assistant', 'Thinking…');
  try {
    const res = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, history: history.slice(-12) }) });
    const data = await res.json(); if (!res.ok) throw new Error(data.detail || 'Request failed');
    pending.remove(); addMessage('assistant', data.answer, data.citations || []);
    history.push({ role: 'user', content: question }, { role: 'assistant', content: data.answer });
  } catch (error) { pending.remove(); addMessage('assistant', `Sorry, ${error.message}`); }
  finally { send.disabled = false; send.textContent = 'Send'; input.focus(); }
});
document.getElementById('question').addEventListener('input', event => { event.target.style.height = 'auto'; event.target.style.height = Math.min(event.target.scrollHeight, 140) + 'px'; });
document.getElementById('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); document.getElementById('chat-form').requestSubmit(); } });
messagesEl.addEventListener('click', event => {
  const suggestion = event.target.closest('[data-question]');
  if (!suggestion) return;
  const input = document.getElementById('question'); input.value = suggestion.dataset.question; input.focus();
  document.getElementById('chat-form').requestSubmit();
});
refreshStatus().catch(() => {});
