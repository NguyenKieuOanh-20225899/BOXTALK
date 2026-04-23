const state = {
  documents: [],
  selectedDocId: null,
  activeCitation: null,
  asking: false,
  uploading: false,
};

const els = {
  health: document.getElementById("healthStatus"),
  uploadForm: document.getElementById("uploadForm"),
  uploadStatus: document.getElementById("uploadStatus"),
  pdfFile: document.getElementById("pdfFile"),
  fileName: document.getElementById("fileName"),
  buildDense: document.getElementById("buildDense"),
  refreshDocs: document.getElementById("refreshDocs"),
  docList: document.getElementById("docList"),
  docCount: document.getElementById("docCount"),
  activeDoc: document.getElementById("activeDoc"),
  askForm: document.getElementById("askForm"),
  questionInput: document.getElementById("questionInput"),
  answerText: document.getElementById("answerText"),
  answerMetrics: document.getElementById("answerMetrics"),
  citationList: document.getElementById("citationList"),
  citationCount: document.getElementById("citationCount"),
  evidencePanel: document.getElementById("evidencePanel"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const ms = Number(value);
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.round(ms)}ms`;
}

function formatDate(value) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("vi-VN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const payload = await res.json();
      detail = payload.detail || detail;
    } catch {
      detail = await res.text();
    }
    throw new Error(detail);
  }
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return null;
  return res.json();
}

async function checkHealth() {
  try {
    await api("/health");
    els.health.textContent = "API ready";
    els.health.className = "health ok";
  } catch {
    els.health.textContent = "API lỗi";
    els.health.className = "health error";
  }
}

async function loadDocuments() {
  state.documents = await api("/documents");
  if (!state.selectedDocId && state.documents.length) {
    const firstReady = state.documents.find((doc) => doc.status === "ready");
    state.selectedDocId = (firstReady || state.documents[0]).doc_id;
  }
  renderDocuments();
  renderActiveDocument();
}

function selectedDocument() {
  return state.documents.find((doc) => doc.doc_id === state.selectedDocId) || null;
}

function renderDocuments() {
  els.docCount.textContent = String(state.documents.length);
  if (!state.documents.length) {
    els.docList.innerHTML = `<div class="empty-state">Chưa có PDF</div>`;
    return;
  }

  els.docList.innerHTML = state.documents
    .map((doc) => {
      const active = doc.doc_id === state.selectedDocId ? " active" : "";
      const statusClass = doc.status === "error" ? " error" : doc.status === "processing" ? " processing" : "";
      const canAsk = doc.status === "ready";
      const pdfLink = doc.pdf_url ? `${doc.pdf_url}#page=1` : "#";
      return `
        <article class="doc-row${active}" data-doc-id="${escapeHtml(doc.doc_id)}">
          <button class="doc-select" type="button" data-select-doc="${escapeHtml(doc.doc_id)}">
            <span class="doc-main">
              <span class="doc-title">${escapeHtml(doc.filename)}</span>
              <span class="status-badge${statusClass}">${escapeHtml(doc.status)}</span>
            </span>
            <span class="doc-meta">
              <span>${escapeHtml(doc.document_type)}</span>
              <span>${escapeHtml(doc.index_mode)}</span>
              <span>${doc.chunk_count || 0} chunks</span>
              <span>${doc.page_count || 0} pages</span>
              <span>${formatMs(doc.build_ms)}</span>
            </span>
          </button>
          <div class="doc-actions">
            <a class="ghost-button" href="${escapeHtml(pdfLink)}" target="_blank" rel="noreferrer">↗ PDF</a>
            <button class="ghost-button" type="button" data-reindex-doc="${escapeHtml(doc.doc_id)}" ${canAsk ? "" : "disabled"}>↻ Re-index</button>
            <button class="danger-button" type="button" data-delete-doc="${escapeHtml(doc.doc_id)}">× Xóa</button>
          </div>
          ${doc.last_error ? `<div class="status-line">${escapeHtml(doc.last_error)}</div>` : ""}
          ${doc.warnings?.length ? `<div class="status-line">${escapeHtml(doc.warnings.join(" "))}</div>` : ""}
        </article>
      `;
    })
    .join("");
}

function renderActiveDocument() {
  const doc = selectedDocument();
  els.activeDoc.textContent = doc ? doc.filename : "Chưa chọn PDF";
  els.questionInput.disabled = !doc || doc.status !== "ready" || state.asking;
  els.askForm.querySelector("button").disabled = !doc || doc.status !== "ready" || state.asking;
}

function renderAnswer(payload) {
  state.activeCitation = null;
  const result = payload?.result;
  if (!result) return;

  els.answerText.textContent = result.answer || "Không có câu trả lời.";
  const evidence = result.evidence_report || {};
  els.answerMetrics.innerHTML = [
    ["route", result.route_action],
    ["type", result.query_type],
    ["strategy", evidence.retrieval_strategy],
    ["latency", formatMs(evidence.total_latency_ms)],
    ["grounded", evidence.grounded],
  ]
    .filter(([, value]) => value !== undefined && value !== null)
    .map(([label, value]) => `<span class="metric-chip">${escapeHtml(label)}: ${escapeHtml(value)}</span>`)
    .join("");

  const hitsById = new Map((result.retrieved_chunks || []).map((hit) => [hit.chunk_id, hit]));
  const citationRows = (result.citations || []).map((citation, index) => {
    const hit = hitsById.get(citation.chunk_id) || {};
    return { citation, hit, index };
  });
  const fallbackRows = !citationRows.length
    ? (result.retrieved_chunks || []).slice(0, 3).map((hit, index) => ({ citation: hit, hit, index }))
    : citationRows;

  els.citationCount.textContent = String(fallbackRows.length);
  els.citationList.innerHTML = fallbackRows.length
    ? fallbackRows
        .map(({ citation, hit, index }) => {
          const page = citation.page || hit.page || "n/a";
          const section = citation.section || hit.section || "No section";
          const score = hit.hybrid_score ?? hit.rerank_score ?? hit.bm25_score ?? "";
          const snippet = hit.text || hit.snippet || "";
          return `
            <button class="citation-card" type="button" data-citation-index="${index}">
              <span class="citation-top">
                <span class="citation-title">Trang ${escapeHtml(page)}</span>
                <span class="citation-score">${score === "" ? "" : Number(score).toFixed(3)}</span>
              </span>
              <span class="doc-meta">
                <span>${escapeHtml(section)}</span>
                <span>${escapeHtml(hit.block_type || "chunk")}</span>
              </span>
              <span class="snippet">${escapeHtml(snippet)}</span>
            </button>
          `;
        })
        .join("")
    : `<div class="empty-state">Không có citation</div>`;

  if (fallbackRows.length) {
    showEvidence(fallbackRows[0]);
  } else {
    renderEmptyEvidence();
  }
  state.lastCitationRows = fallbackRows;
}

function showEvidence(row) {
  const doc = selectedDocument();
  const citation = row.citation || {};
  const hit = row.hit || {};
  const page = citation.page || hit.page || null;
  const section = citation.section || hit.section || "No section";
  const text = hit.text || hit.snippet || "";
  const pageUrl = doc?.pdf_url && page ? `${doc.pdf_url}#page=${page}` : doc?.pdf_url;
  state.activeCitation = row.index;

  els.evidencePanel.innerHTML = `
    <p class="eyebrow">Evidence</p>
    <h3>Trang ${escapeHtml(page || "n/a")}</h3>
    <div class="doc-meta">
      <span>${escapeHtml(section)}</span>
      <span>${escapeHtml(hit.chunk_id || citation.chunk_id || "chunk")}</span>
    </div>
    <p class="evidence-text">${escapeHtml(text || "Không có đoạn trích.")}</p>
    <div class="evidence-actions">
      ${pageUrl ? `<a class="primary-button" href="${escapeHtml(pageUrl)}" target="_blank" rel="noreferrer">↗ Mở PDF</a>` : ""}
    </div>
  `;

  [...els.citationList.querySelectorAll(".citation-card")].forEach((card) => {
    card.classList.toggle("active", Number(card.dataset.citationIndex) === row.index);
  });
}

function renderEmptyEvidence() {
  els.evidencePanel.innerHTML = `
    <p class="eyebrow">Evidence</p>
    <h3>Chưa chọn citation</h3>
    <p class="evidence-text">Citation sẽ hiện đoạn trích, trang và section tại đây.</p>
  `;
}

els.pdfFile.addEventListener("change", () => {
  els.fileName.textContent = els.pdfFile.files[0]?.name || "Chọn PDF";
});

els.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = els.pdfFile.files[0];
  if (!file) {
    els.uploadStatus.textContent = "Chưa chọn PDF";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("build_dense", els.buildDense.checked ? "true" : "false");

  state.uploading = true;
  els.uploadStatus.textContent = "Đang xử lý";
  els.uploadForm.querySelector("button[type='submit']").disabled = true;
  try {
    const doc = await api("/documents", { method: "POST", body: formData });
    state.selectedDocId = doc.doc_id;
    els.uploadStatus.textContent = doc.status === "ready" ? "Đã sẵn sàng hỏi đáp" : `Lỗi: ${doc.last_error || "index failed"}`;
    els.pdfFile.value = "";
    els.fileName.textContent = "Chọn PDF";
    await loadDocuments();
  } catch (err) {
    els.uploadStatus.textContent = `Lỗi: ${err.message}`;
  } finally {
    state.uploading = false;
    els.uploadForm.querySelector("button[type='submit']").disabled = false;
  }
});

els.refreshDocs.addEventListener("click", async () => {
  els.refreshDocs.disabled = true;
  try {
    await loadDocuments();
  } finally {
    els.refreshDocs.disabled = false;
  }
});

els.docList.addEventListener("click", async (event) => {
  const selectButton = event.target.closest("[data-select-doc]");
  const deleteButton = event.target.closest("[data-delete-doc]");
  const reindexButton = event.target.closest("[data-reindex-doc]");

  if (selectButton) {
    state.selectedDocId = selectButton.dataset.selectDoc;
    renderDocuments();
    renderActiveDocument();
    return;
  }

  if (deleteButton) {
    const docId = deleteButton.dataset.deleteDoc;
    const doc = state.documents.find((item) => item.doc_id === docId);
    if (!confirm(`Xóa ${doc?.filename || docId}?`)) return;
    await api(`/documents/${docId}`, { method: "DELETE" });
    if (state.selectedDocId === docId) state.selectedDocId = null;
    await loadDocuments();
    return;
  }

  if (reindexButton) {
    const docId = reindexButton.dataset.reindexDoc;
    const doc = state.documents.find((item) => item.doc_id === docId);
    reindexButton.disabled = true;
    try {
      await api(`/documents/${docId}/reindex`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ build_dense: doc?.index_mode === "hybrid" }),
      });
      await loadDocuments();
    } finally {
      reindexButton.disabled = false;
    }
  }
});

els.askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const doc = selectedDocument();
  const question = els.questionInput.value.trim();
  if (!doc || doc.status !== "ready" || !question) return;

  state.asking = true;
  renderActiveDocument();
  els.answerText.textContent = "Đang truy hồi evidence";
  els.answerMetrics.innerHTML = "";
  els.citationList.innerHTML = "";
  els.citationCount.textContent = "0";
  renderEmptyEvidence();

  try {
    const payload = await api(`/documents/${doc.doc_id}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    renderAnswer(payload);
  } catch (err) {
    els.answerText.textContent = `Lỗi: ${err.message}`;
  } finally {
    state.asking = false;
    renderActiveDocument();
  }
});

els.citationList.addEventListener("click", (event) => {
  const card = event.target.closest("[data-citation-index]");
  if (!card || !state.lastCitationRows) return;
  const row = state.lastCitationRows[Number(card.dataset.citationIndex)];
  if (row) showEvidence(row);
});

checkHealth();
loadDocuments().catch((err) => {
  els.docList.innerHTML = `<div class="empty-state">${escapeHtml(err.message)}</div>`;
});
