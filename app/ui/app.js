const state = {
  documents: [],
  selectedDocId: null,
  activeCitationIndex: 0,
  citationRows: [],
  lastPayload: null,
  asking: false,
  uploading: false,
  pendingUploadFile: null,
  developerMode: window.localStorage.getItem("boxtalk.developerMode") === "1",
};

const els = {
  health: document.getElementById("healthStatus"),
  libraryStats: document.getElementById("libraryStats"),
  developerModeToggle: document.getElementById("developerModeToggle"),
  uploadForm: document.getElementById("uploadForm"),
  uploadDropzone: document.getElementById("uploadDropzone"),
  uploadStatus: document.getElementById("uploadStatus"),
  pdfFile: document.getElementById("pdfFile"),
  selectedFileMeta: document.getElementById("selectedFileMeta"),
  buildDense: document.getElementById("buildDense"),
  refreshDocs: document.getElementById("refreshDocs"),
  docList: document.getElementById("docList"),
  docCount: document.getElementById("docCount"),
  activeDocName: document.getElementById("activeDocName"),
  activeDocMeta: document.getElementById("activeDocMeta"),
  askForm: document.getElementById("askForm"),
  askButton: document.getElementById("askButton"),
  questionInput: document.getElementById("questionInput"),
  answerText: document.getElementById("answerText"),
  answerNote: document.getElementById("answerNote"),
  answerBadges: document.getElementById("answerBadges"),
  citationCount: document.getElementById("citationCount"),
  citationList: document.getElementById("citationList"),
  sourceViewer: document.getElementById("sourceViewer"),
  developerPanel: document.getElementById("developerPanel"),
  developerSummary: document.getElementById("developerSummary"),
  routeAttemptList: document.getElementById("routeAttemptList"),
  rawTrace: document.getElementById("rawTrace"),
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
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function toLabel(value) {
  if (!value) return "Unknown";
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function formatStatus(status) {
  const normalized = String(status || "unknown").toLowerCase();
  if (normalized === "ready") return "Ready";
  if (normalized === "processing") return "Indexing";
  if (normalized === "error") return "Error";
  if (normalized === "uploaded") return "Uploaded";
  return toLabel(normalized);
}

function statusTone(status) {
  const normalized = String(status || "unknown").toLowerCase();
  if (normalized === "ready") return "ready";
  if (normalized === "processing") return "processing";
  if (normalized === "error") return "error";
  return "uploaded";
}

function buildPageUrl(doc, page) {
  if (!doc?.pdf_url) return null;
  if (!page) return doc.pdf_url;
  return `${doc.pdf_url}#page=${page}`;
}

function pickScore(hit) {
  if (!hit) return null;
  const candidates = [hit.hybrid_score, hit.rerank_score, hit.bm25_score, hit.dense_score];
  for (const value of candidates) {
    if (value !== null && value !== undefined && value !== "") return Number(value);
  }
  return null;
}

function selectedDocument() {
  return state.documents.find((doc) => doc.doc_id === state.selectedDocId) || null;
}

function getPendingFile() {
  return state.pendingUploadFile || els.pdfFile.files[0] || null;
}

function setPendingFile(file) {
  state.pendingUploadFile = file || null;
  const label = file ? `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB` : "PDF only";
  els.selectedFileMeta.textContent = label;
}

function clearPendingFile() {
  state.pendingUploadFile = null;
  els.pdfFile.value = "";
  els.selectedFileMeta.textContent = "PDF only";
}

function setDeveloperMode(enabled) {
  state.developerMode = Boolean(enabled);
  window.localStorage.setItem("boxtalk.developerMode", state.developerMode ? "1" : "0");
  document.body.classList.toggle("dev-mode", state.developerMode);
  els.developerModeToggle.checked = state.developerMode;
  renderDocuments();
  renderActiveDocument();
  if (state.lastPayload) {
    renderAnswer(state.lastPayload, { preserveSelection: true });
  } else {
    renderDeveloperPanel(null);
    renderSourceViewer(null);
  }
}

function setUploadStatus(message, tone = "neutral") {
  els.uploadStatus.textContent = message;
  els.uploadStatus.dataset.tone = tone;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return null;
  return response.json();
}

async function checkHealth() {
  try {
    await api("/health");
    els.health.textContent = "API ready";
    els.health.className = "summary-pill ok";
  } catch {
    els.health.textContent = "API unavailable";
    els.health.className = "summary-pill error";
  }
}

function updateLibraryStats() {
  const total = state.documents.length;
  const ready = state.documents.filter((doc) => doc.status === "ready").length;
  const processing = state.documents.filter((doc) => doc.status === "processing").length;
  els.libraryStats.textContent = `${ready}/${total} ready`;
  if (processing > 0) {
    els.libraryStats.textContent += ` · ${processing} indexing`;
  }
  els.docCount.textContent = String(total);
}

async function loadDocuments() {
  const previousSelected = state.selectedDocId;
  state.documents = await api("/documents");
  const ids = new Set(state.documents.map((doc) => doc.doc_id));
  if (!ids.has(state.selectedDocId)) {
    const firstReady = state.documents.find((doc) => doc.status === "ready");
    state.selectedDocId = (firstReady || state.documents[0] || {}).doc_id || null;
    if (previousSelected && previousSelected !== state.selectedDocId) {
      resetQAView();
    }
  }
  updateLibraryStats();
  renderDocuments();
  renderActiveDocument();
}

function renderDocuments() {
  updateLibraryStats();
  if (!state.documents.length) {
    els.docList.innerHTML = `<div class="empty-state">No PDFs uploaded yet.</div>`;
    return;
  }

  els.docList.innerHTML = state.documents
    .map((doc) => {
      const active = doc.doc_id === state.selectedDocId ? " active" : "";
      const docUrl = buildPageUrl(doc, 1) || "#";
      const ready = doc.status === "ready";
      const warnings = Array.isArray(doc.warnings) ? doc.warnings : [];
      return `
        <article class="doc-card${active}">
          <div class="doc-card-head">
            <div>
              <button class="doc-title-button" type="button" data-select-doc="${escapeHtml(doc.doc_id)}">${escapeHtml(doc.filename)}</button>
              <div class="doc-secondary">${escapeHtml(toLabel(doc.document_type))} · ${escapeHtml(String(doc.page_count || 0))} pages</div>
            </div>
            <span class="status-chip ${statusTone(doc.status)}">${escapeHtml(formatStatus(doc.status))}</span>
          </div>

          <div class="meta-row">
            <span class="meta-pill">${escapeHtml(toLabel(doc.document_type))}</span>
            <span class="meta-pill">${escapeHtml(String(doc.page_count || 0))} pages</span>
            <span class="meta-pill dev-only">${escapeHtml(String(doc.chunk_count || 0))} chunks</span>
            <span class="meta-pill dev-only">${escapeHtml(doc.index_mode || "bm25")}</span>
            <span class="meta-pill dev-only">${escapeHtml(formatMs(doc.build_ms))}</span>
          </div>

          ${
            doc.last_error
              ? `<div class="status-line">${escapeHtml(doc.last_error)}</div>`
              : warnings.length
                ? `<div class="status-line">${escapeHtml(warnings.join(" "))}</div>`
                : `<div class="status-line">Indexed ${escapeHtml(formatDate(doc.indexed_at || doc.created_at))}</div>`
          }

          <div class="card-actions">
            <button class="secondary-button" type="button" data-chat-doc="${escapeHtml(doc.doc_id)}" ${ready ? "" : "disabled"}>Open chat</button>
            <a class="ghost-button" href="${escapeHtml(docUrl)}" target="_blank" rel="noreferrer">Open PDF</a>
            <button class="ghost-button" type="button" data-reindex-doc="${escapeHtml(doc.doc_id)}">Re-index</button>
            <button class="danger-button" type="button" data-delete-doc="${escapeHtml(doc.doc_id)}">Delete</button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderActiveDocument() {
  const doc = selectedDocument();
  const canAsk = Boolean(doc && doc.status === "ready" && !state.asking);
  els.questionInput.disabled = !canAsk;
  els.askButton.disabled = !canAsk;

  if (!doc) {
    els.activeDocName.textContent = "No document selected";
    els.activeDocMeta.textContent = "Choose a ready PDF to start grounded QA.";
    els.questionInput.placeholder = "Ask a question about the selected PDF";
    return;
  }

  els.activeDocName.textContent = doc.filename;
  els.activeDocMeta.textContent = `${toLabel(doc.document_type)} · ${doc.page_count || 0} pages · ${formatStatus(doc.status)}`;
  els.questionInput.placeholder =
    doc.status === "ready" ? `Ask a question about ${doc.filename}` : "Wait until indexing is complete";
}

function renderAnswerBadges(result, citationCount) {
  const badges = [];
  const routeAction = String(result?.route_action || "");
  const finalSource = String(result?.final_answer_source || "standard");
  const grounded = Boolean(result?.grounded);
  const fallbackUsed = finalSource !== "standard";

  if (grounded && routeAction === "answer") {
    badges.push(`<span class="badge success">Grounded</span>`);
  } else if (routeAction === "abstain") {
    badges.push(`<span class="badge warn">Insufficient evidence</span>`);
  } else {
    badges.push(`<span class="badge warn">Weak evidence</span>`);
  }

  if (citationCount > 0) {
    badges.push(`<span class="badge neutral">${escapeHtml(String(citationCount))} source${citationCount === 1 ? "" : "s"}</span>`);
  }

  if (fallbackUsed) {
    badges.push(`<span class="badge info">Enhanced reasoning</span>`);
  }

  return badges.join("");
}

function buildAnswerNote(result, citationCount) {
  const routeAction = String(result?.route_action || "");
  if (routeAction === "abstain") {
    return "The current evidence is not sufficient to support a confident answer. Review the source panel or upload a more relevant document.";
  }
  if (!result?.grounded) {
    return "Related evidence was found, but the answer should be reviewed against the cited passages.";
  }
  if (citationCount === 0) {
    return "The answer is grounded, but no explicit citation card was emitted. Review the retrieved evidence.";
  }
  return `Grounded answer backed by ${citationCount} cited source${citationCount === 1 ? "" : "s"}.`;
}

function buildCitationRows(result) {
  const hitsById = new Map((result?.retrieved_chunks || []).map((hit) => [hit.chunk_id, hit]));
  const citedRows = (result?.citations || []).map((citation, index) => ({
    index,
    citation,
    hit: hitsById.get(citation.chunk_id) || {},
    citationKind: "citation",
  }));

  if (citedRows.length) return citedRows;

  return (result?.retrieved_chunks || []).slice(0, 3).map((hit, index) => ({
    index,
    citation: hit,
    hit,
    citationKind: "retrieved_hit",
  }));
}

function renderCitationList(rows) {
  const doc = selectedDocument();
  els.citationCount.textContent = String(rows.length);
  if (!rows.length) {
    els.citationList.innerHTML = `<div class="empty-state">No citations available for the current answer.</div>`;
    return;
  }

  els.citationList.innerHTML = rows
    .map((row) => {
      const citation = row.citation || {};
      const hit = row.hit || {};
      const page = citation.page || hit.page || "n/a";
      const pageUrl = buildPageUrl(doc, page === "n/a" ? null : page);
      const section = citation.section || hit.section || "No section";
      const blockType = hit.block_type || citation.block_type || "chunk";
      const score = pickScore(hit);
      const snippet = hit.text || hit.snippet || citation.snippet || "";
      const active = row.index === state.activeCitationIndex ? " active" : "";
      return `
        <article class="citation-card${active}">
          <div class="citation-top">
            <div>
              <div class="citation-title">Page ${escapeHtml(page)}</div>
              <div class="doc-secondary">${escapeHtml(section)}</div>
            </div>
            <div class="score-label">${score === null ? "" : score.toFixed(3)}</div>
          </div>

          <div class="meta-row">
            <span class="meta-pill">${escapeHtml(toLabel(blockType))}</span>
            <span class="meta-pill">${row.citationKind === "citation" ? "Cited evidence" : "Top retrieved chunk"}</span>
            <span class="meta-pill dev-only">${escapeHtml(hit.chunk_id || citation.chunk_id || "n/a")}</span>
          </div>

          <div class="snippet">${escapeHtml(snippet || "No snippet available.")}</div>

          <div class="card-actions">
            <button class="secondary-button" type="button" data-select-citation="${row.index}">View source</button>
            ${
              pageUrl
                ? `<a class="ghost-button" href="${escapeHtml(pageUrl)}" target="_blank" rel="noreferrer">Open page</a>`
                : `<button class="ghost-button" type="button" disabled>Open page</button>`
            }
          </div>
        </article>
      `;
    })
    .join("");
}

function renderSourceViewer(row) {
  if (!row) {
    els.sourceViewer.innerHTML = `
      <div class="empty-state">
        Select a source card to inspect the cited snippet, page number, and source actions.
      </div>
    `;
    return;
  }

  const doc = selectedDocument();
  const citation = row.citation || {};
  const hit = row.hit || {};
  const page = citation.page || hit.page || "n/a";
  const section = citation.section || hit.section || "No section";
  const blockType = hit.block_type || citation.block_type || "chunk";
  const chunkId = hit.chunk_id || citation.chunk_id || "n/a";
  const snippet = hit.text || hit.snippet || citation.snippet || "No snippet available.";
  const pageUrl = buildPageUrl(doc, page === "n/a" ? null : page);
  const docUrl = doc?.pdf_url || null;
  const score = pickScore(hit);

  els.sourceViewer.innerHTML = `
    <div class="source-header">
      <p class="eyebrow">Source</p>
      <h3>Page ${escapeHtml(page)}</h3>
      <div class="source-subtitle">${escapeHtml(doc?.filename || hit.source_name || "Selected source")}</div>
    </div>

    <div class="meta-row">
      <span class="meta-pill">${escapeHtml(section)}</span>
      <span class="meta-pill">${escapeHtml(toLabel(blockType))}</span>
      ${score === null ? "" : `<span class="meta-pill dev-only">score ${escapeHtml(score.toFixed(3))}</span>`}
      <span class="meta-pill dev-only">${escapeHtml(chunkId)}</span>
    </div>

    <div class="source-snippet">${escapeHtml(snippet)}</div>

    <div class="source-actions">
      ${
        pageUrl
          ? `<a class="primary-button" href="${escapeHtml(pageUrl)}" target="_blank" rel="noreferrer">Open page</a>`
          : ""
      }
      ${
        docUrl
          ? `<a class="ghost-button" href="${escapeHtml(docUrl)}" target="_blank" rel="noreferrer">Open document</a>`
          : ""
      }
    </div>
  `;

  for (const card of els.citationList.querySelectorAll(".citation-card")) {
    const button = card.querySelector("[data-select-citation]");
    const index = Number(button?.dataset.selectCitation);
    card.classList.toggle("active", index === row.index);
  }
}

function renderDeveloperPanel(payload) {
  if (!state.developerMode) {
    els.developerPanel.hidden = true;
    return;
  }

  const result = payload?.result;
  if (!result) {
    els.developerPanel.hidden = false;
    els.developerSummary.innerHTML = `<div class="empty-state">Ask a question to inspect route and fallback details.</div>`;
    els.routeAttemptList.innerHTML = "";
    els.rawTrace.textContent = "";
    return;
  }

  const evidence = result.evidence_report || {};
  const fallback = result.fallback_trace || evidence.fallback_trace || {};
  const summaryRows = [
    ["Route action", result.route_action],
    ["Query type", result.query_type],
    ["Retrieval", evidence.retrieval_strategy],
    ["Latency", formatMs(result.latency_ms || evidence.total_latency_ms)],
    ["Grounded", String(Boolean(result.grounded))],
    ["Final source", result.final_answer_source],
    ["Fallback called", String(Boolean(fallback.fallback_called ?? fallback.called))],
    ["Fallback used", String(Boolean(fallback.fallback_used ?? fallback.used))],
    ["Reasoning mode", fallback.reasoning_mode || "n/a"],
    ["Provider", fallback.provider_name || "n/a"],
    ["Override confidence", fallback.override_confidence ?? fallback.confidence ?? "n/a"],
    ["Evidence sufficiency", evidence.sufficiency ?? "n/a"],
  ];

  els.developerSummary.innerHTML = summaryRows
    .map(
      ([key, value]) => `
        <div class="kv-row">
          <div class="kv-key">${escapeHtml(key)}</div>
          <div class="kv-value">${escapeHtml(String(value ?? "n/a"))}</div>
        </div>
      `,
    )
    .join("");

  const routeAttempts = Array.isArray(result.route_attempts) ? result.route_attempts : [];
  els.routeAttemptList.innerHTML = routeAttempts.length
    ? routeAttempts
        .map((attempt) => {
          const selected = attempt.selected ? " selected" : "";
          return `
            <article class="route-card${selected}">
              <div class="route-top">
                <div class="route-label">Attempt ${escapeHtml(String(attempt.attempt_index ?? 0))}</div>
                <span class="status-chip ${attempt.selected ? "ready" : "uploaded"}">${attempt.selected ? "Selected" : "Candidate"}</span>
              </div>
              <div class="meta-row">
                <span class="meta-pill">${escapeHtml(String(attempt.query_type || "n/a"))}</span>
                <span class="meta-pill">${escapeHtml(String(attempt.retrieval_strategy || "n/a"))}</span>
                <span class="meta-pill">${escapeHtml(String(attempt.evidence_decision || "n/a"))}</span>
              </div>
              <div class="route-copy">
                retry: ${escapeHtml(String(attempt.retry_reason || "n/a"))}<br />
                quality: ${escapeHtml(String(attempt.quality_score ?? "n/a"))}<br />
                fallback: ${escapeHtml(String(Boolean(attempt.fallback_used)))}
              </div>
            </article>
          `;
        })
        .join("")
    : `<div class="empty-state">No route attempt trace available.</div>`;

  els.rawTrace.textContent = JSON.stringify(
    {
      fallback_trace: fallback,
      evidence_report: evidence,
      route_attempts: routeAttempts,
      standard_answer: result.standard_answer,
    },
    null,
    2,
  );

  els.developerPanel.hidden = false;
}

function resetQAView() {
  state.lastPayload = null;
  state.citationRows = [];
  state.activeCitationIndex = 0;
  els.answerBadges.innerHTML = "";
  els.answerText.textContent = "Choose a ready document and ask a question.";
  els.answerNote.textContent = "The answer card highlights groundedness first. Developer traces stay behind the toggle.";
  els.citationCount.textContent = "0";
  els.citationList.innerHTML = `<div class="empty-state">No sources yet.</div>`;
  renderSourceViewer(null);
  renderDeveloperPanel(null);
}

function renderAnswer(payload, options = {}) {
  state.lastPayload = payload;
  const result = payload?.result;
  if (!result) {
    resetQAView();
    return;
  }

  const rows = buildCitationRows(result);
  state.citationRows = rows;

  if (!options.preserveSelection || state.activeCitationIndex >= rows.length) {
    state.activeCitationIndex = 0;
  }

  els.answerText.textContent = result.answer || "No answer available.";
  els.answerBadges.innerHTML = renderAnswerBadges(result, rows.length);
  els.answerNote.textContent = buildAnswerNote(result, rows.length);
  renderCitationList(rows);
  renderSourceViewer(rows[state.activeCitationIndex] || null);
  renderDeveloperPanel(payload);
}

function renderErrorAnswer(message) {
  state.lastPayload = null;
  state.citationRows = [];
  state.activeCitationIndex = 0;
  els.answerBadges.innerHTML = `<span class="badge danger">Request failed</span>`;
  els.answerText.textContent = `Error: ${message}`;
  els.answerNote.textContent = "The system could not complete the request. Check the API state and document status.";
  els.citationCount.textContent = "0";
  els.citationList.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
  renderSourceViewer(null);
  renderDeveloperPanel(null);
}

function focusQuestionInput() {
  if (!els.questionInput.disabled) {
    els.questionInput.focus();
  }
}

els.developerModeToggle.addEventListener("change", () => {
  setDeveloperMode(els.developerModeToggle.checked);
});

els.pdfFile.addEventListener("change", () => {
  setPendingFile(els.pdfFile.files[0] || null);
});

["dragenter", "dragover"].forEach((eventName) => {
  els.uploadDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.uploadDropzone.classList.add("dragover");
  });
});

["dragleave", "dragend", "drop"].forEach((eventName) => {
  els.uploadDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.uploadDropzone.classList.remove("dragover");
  });
});

els.uploadDropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  setPendingFile(file);
});

els.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = getPendingFile();
  if (!file) {
    setUploadStatus("Choose a PDF before uploading.", "warn");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("build_dense", els.buildDense.checked ? "true" : "false");

  state.uploading = true;
  setUploadStatus("Indexing document...", "info");
  els.uploadForm.querySelector("button[type='submit']").disabled = true;

  try {
    const documentInfo = await api("/documents", {
      method: "POST",
      body: formData,
    });
    state.selectedDocId = documentInfo.doc_id;
    clearPendingFile();
    setUploadStatus(
      documentInfo.status === "ready" ? "Document indexed and ready for grounded QA." : `Indexing failed: ${documentInfo.last_error || "unknown error"}`,
      documentInfo.status === "ready" ? "success" : "danger",
    );
    await loadDocuments();
    resetQAView();
    focusQuestionInput();
  } catch (error) {
    setUploadStatus(`Upload failed: ${error.message}`, "danger");
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
  const chatButton = event.target.closest("[data-chat-doc]");
  const deleteButton = event.target.closest("[data-delete-doc]");
  const reindexButton = event.target.closest("[data-reindex-doc]");

  if (selectButton || chatButton) {
    const docId = (selectButton || chatButton).dataset.selectDoc || (selectButton || chatButton).dataset.chatDoc;
    if (docId !== state.selectedDocId) {
      state.selectedDocId = docId;
      resetQAView();
    }
    renderDocuments();
    renderActiveDocument();
    focusQuestionInput();
    return;
  }

  if (deleteButton) {
    const docId = deleteButton.dataset.deleteDoc;
    const doc = state.documents.find((item) => item.doc_id === docId);
    if (!window.confirm(`Delete ${doc?.filename || docId}?`)) return;
    await api(`/documents/${docId}`, { method: "DELETE" });
    if (state.selectedDocId === docId) {
      state.selectedDocId = null;
      resetQAView();
    }
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
      if (state.selectedDocId === docId) {
        resetQAView();
      }
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
  els.answerBadges.innerHTML = `<span class="badge info">Retrieving evidence</span>`;
  els.answerText.textContent = "Retrieving evidence and building a grounded answer...";
  els.answerNote.textContent = "This can take a little longer when fallback reasoning is triggered.";
  els.citationCount.textContent = "0";
  els.citationList.innerHTML = `<div class="empty-state">Waiting for citations...</div>`;
  renderSourceViewer(null);
  renderDeveloperPanel(null);

  try {
    const payload = await api(`/documents/${doc.doc_id}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    renderAnswer(payload);
  } catch (error) {
    renderErrorAnswer(error.message);
  } finally {
    state.asking = false;
    renderActiveDocument();
  }
});

els.citationList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-select-citation]");
  if (!button) return;
  const index = Number(button.dataset.selectCitation);
  if (!Number.isFinite(index) || !state.citationRows[index]) return;
  state.activeCitationIndex = index;
  renderSourceViewer(state.citationRows[index]);
});

setDeveloperMode(state.developerMode);
checkHealth();
loadDocuments().catch((error) => {
  els.docList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  renderActiveDocument();
});
