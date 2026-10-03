/* ==========================================================================
   app.js - front-end logic for the merged Fact Check app.
   Talks to the FastAPI backend it is served from (same origin), so there is
   nothing to configure: the UI and the API share one server and one port.
   ========================================================================== */
"use strict";

const API_BASE =
  location.protocol === "file:" ? "http://127.0.0.1:8000" : location.origin;

const MAX_BYTES = 10 * 1024 * 1024; // mirrors MAX_IMAGE_BYTES on the backend
const OK_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

/* ------------------------------ samples -------------------------------- */
const SAMPLES = {
  real:
    "A study published in the journal Nature Climate Change found that global " +
    "average sea surface temperatures in 2024 were the highest recorded since " +
    "instrumental measurements began in 1850. The research was led by NOAA and " +
    "independently confirmed by NASA and the UK Met Office Hadley Centre.",
  partial:
    "The Indian government has announced that from next month every citizen will " +
    "receive free unlimited 5G internet on their mobile phone. Officials said the " +
    "scheme is fully funded and will be implemented nationwide within a week.",
  fake:
    "BREAKING: NASA has confirmed that the Earth will experience 15 days of " +
    "complete darkness in December because of a rare alignment of Jupiter and " +
    "Saturn. Forward this message to 10 people to stay safe.",
};

/* -------------------------------- DOM ---------------------------------- */
const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const dzEmpty = $("dzEmpty");
const dzPreview = $("dzPreview");
const dzThumb = $("dzThumb");
const dzName = $("dzName");
const dzSize = $("dzSize");
const dzRemove = $("dzRemove");
const articleText = $("articleText");
const wordCount = $("wordCount");
const checkBtn = $("checkBtn");
const ctaLabel = $("ctaLabel");
const ctaSpinner = $("ctaSpinner");
const ctaNote = $("ctaNote");
const results = $("results");

let selectedFile = null;
let previewUrl = null;

/* ------------------------------ helpers -------------------------------- */
function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function hostOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch (_) {
    return "";
  }
}

function humanSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function note(message) {
  if (!message) {
    ctaNote.hidden = true;
    ctaNote.textContent = "";
    return;
  }
  ctaNote.textContent = message;
  ctaNote.hidden = false;
}

/* ---------------------------- image handling --------------------------- */
function setFile(file) {
  if (!file) return;
  const type = (file.type || "").toLowerCase();
  if (!OK_TYPES.includes(type)) {
    note("Unsupported file type. Upload a PNG, JPG or WebP image.");
    return;
  }
  if (file.size > MAX_BYTES) {
    note("That image is " + humanSize(file.size) + "; the limit is 10 MB.");
    return;
  }
  note("");
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  dzThumb.src = previewUrl;
  dzName.textContent = file.name || "pasted-image.png";
  dzSize.textContent = humanSize(file.size) + " \u00b7 " + type;
  dzEmpty.hidden = true;
  dzPreview.hidden = false;
}

function clearFile() {
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  fileInput.value = "";
  dzThumb.removeAttribute("src");
  dzEmpty.hidden = false;
  dzPreview.hidden = true;
}

/* --------------------------- drag & drop ------------------------------- */
["dragenter", "dragover"].forEach((event) =>
  dropzone.addEventListener(event, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-over");
  })
);
["dragleave", "drop"].forEach((event) =>
  dropzone.addEventListener(event, () => dropzone.classList.remove("is-over"))
);

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length) setFile(files[0]);
});

dropzone.addEventListener("click", (e) => {
  if (e.target === dzRemove || dzRemove.contains(e.target)) return;
  if (selectedFile) return;
  fileInput.click();
});
dropzone.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !selectedFile) {
    e.preventDefault();
    fileInput.click();
  }
});

$("browseBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files.length) setFile(fileInput.files[0]);
});
dzRemove.addEventListener("click", (e) => {
  e.stopPropagation();
  clearFile();
});

/* Ctrl+V anywhere on the page pastes a screenshot straight into the box. */
window.addEventListener("paste", (e) => {
  const items = (e.clipboardData && e.clipboardData.items) || [];
  for (const item of items) {
    if (item.kind === "file" && item.type.startsWith("image/")) {
      const blob = item.getAsFile();
      if (blob) {
        const ext = (blob.type.split("/")[1] || "png").replace("jpeg", "jpg");
        setFile(new File([blob], "pasted-screenshot." + ext, { type: blob.type }));
        e.preventDefault();
      }
      return;
    }
  }
});
/* ---------------------------- word counter ----------------------------- */
function updateWordCount() {
  const words = articleText.value.trim().split(/\s+/).filter(Boolean).length;
  wordCount.textContent = words + (words === 1 ? " word" : " words");
}

/* ------------------------------ verdict UI ----------------------------- */
const VERDICTS = {
  TRUE: {
    cls: "v-true", emoji: "\u2705", label: "TRUE / GENUINE", accent: "#16a34a",
    blurb: "The claim is corroborated by reliable sources.",
    sub: "\u2713 Corroborated by reliable, independent sources.",
  },
  FALSE: {
    cls: "v-false", emoji: "\u274c", label: "FALSE / FAKE", accent: "#dc2626",
    blurb: "The claim is contradicted by reliable sources or fabricated.",
    sub: "\u2715 Contradicted by reliable sources or fabricated.",
  },
  PARTIALLY_TRUE: {
    cls: "v-partial", emoji: "\u26a0\ufe0f", label: "PARTIALLY TRUE / MISLEADING",
    accent: "#d97706", blurb: "Parts of the claim are accurate; key parts are not.",
    sub: "\u26a0 Mixed: some assertions hold, key ones do not.",
  },
};
const UNKNOWN_VERDICT = {
  cls: "v-unknown", emoji: "\u2753", label: "UNKNOWN", accent: "#6b7280",
  blurb: "", sub: "The verdict could not be determined.",
};

function showLoading(imageAttached) {
  results.hidden = false;
  results.innerHTML =
    '<div class="loading-card">' +
    "<strong>Checking the claim\u2026</strong>" +
    "<p>Searching the live web" +
    (imageAttached ? " and inspecting the screenshot" : "") +
    " and cross-referencing reliable sources.</p>" +
    '<div class="bar"><span></span></div>' +
    "</div>";
  results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(message) {
  results.hidden = false;
  results.innerHTML =
    '<div class="result-card">' +
    '<div class="verdict-banner v-false">' +
    '<span class="v-emoji">\u26a0\ufe0f</span>' +
    "<div><div class=\"v-label\">Check could not be completed</div>" +
    '<div class="v-blurb">' + escapeHtml(message) + "</div></div>" +
    "</div></div>";
}

function sourcesHtml(sources) {
  const items = (sources || []).filter((s) => s && s.url);
  if (!items.length) {
    return (
      '<p class="no-sources">No source URLs were returned. This happens when the ' +
      "live web search quota is exhausted, so the verdict was produced from the " +
      "model's own knowledge only. Treat it as unverified.</p>"
    );
  }
  return (
    '<ul class="sources">' +
    items
      .map((source, index) => {
        const url = String(source.url);
        const title = source.title || hostOf(url) || url;
        const host = hostOf(url);
        return (
          "<li><span class=\"num\">" + (index + 1) + "</span><div>" +
          '<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener noreferrer">' +
          escapeHtml(title) + "</a>" +
          (host ? '<span class="host">' + escapeHtml(host) + "</span>" : "") +
          "</div></li>"
        );
      })
      .join("") +
    "</ul>"
function renderResult(payload, imageAttached) {
  const verdictKey = String(payload.verdict || "").toUpperCase();
  const style = VERDICTS[verdictKey] || UNKNOWN_VERDICT;
  let score = Number(payload.truth_percentage);
  if (!isFinite(score)) score = 0;
  score = Math.max(0, Math.min(100, score));

  const analysis = String(payload.analysis || "").trim();
  const details = String(payload.details || "").trim();
  const detailsTitle =
    verdictKey === "TRUE" ? "\ud83d\udcda Verified context" : "\u2696\ufe0f Factual correction";

  results.hidden = false;
    '<div class="v-sub">' + escapeHtml(style.sub) + "</div>" +
  results.innerHTML =
    '<div class="result-card">' +
    '<div class="verdict-banner ' + style.cls + '">' +
    '<span class="v-emoji">' + style.emoji + "</span><div>" +
    '<div class="v-label">' + escapeHtml(style.label) + "</div>" +
    '<div class="v-blurb">' + escapeHtml(style.blurb) + "</div>" +
    "</div></div>" +
    '<div class="score-row">' +
    '<div><div class="score-caption">Truth score</div>' +
    '<div class="score-value" style="color:' + style.bar + '">' + Math.round(score) + "%</div></div>" +
    "</div>" +
    '<div class="result-body">' +
    '<div><h3 class="block-title">\ud83e\udde0 Analysis</h3>' +
    '<p class="prose' + (analysis ? "" : " is-empty") + '">' +
    escapeHtml(analysis || "No analysis was returned.") + "</p></div>" +
    '<div><h3 class="block-title">' + detailsTitle + "</h3>" +
    '<p class="prose' + (details ? "" : " is-empty") + '">' +
    escapeHtml(details || "No extra detail was returned.") + "</p></div>" +
    '<div><h3 class="block-title">\ud83d\udd17 Sources</h3>' +
    sourcesHtml(payload.sources) + "</div>" +
    "</div>" +
    '<details class="raw"><summary>\ud83e\uddfe Raw JSON response</summary>' +
    "<pre>" + escapeHtml(JSON.stringify(payload, null, 2)) + "</pre></details>" +
    "</div>";

  if (imageAttached) note("");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}
/* ------------------------------ API calls ------------------------------ */
async function readError(response) {
  try {
    const body = await response.json();
    const detail = body.detail || body.error || body;
    if (Array.isArray(detail)) {
      return detail.map((d) => (d && d.msg) || JSON.stringify(d)).join("; ");
    }
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch (_) {
    return "HTTP " + response.status + " " + response.statusText;
  }
}

async function verifyText(text) {
  const response = await fetch(API_BASE + "/verify/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

async function verifyImage(file, claim) {
  const form = new FormData();
  form.append("file", file, file.name);
  if (claim && claim.trim()) form.append("claim", claim.trim());
  const response = await fetch(API_BASE + "/verify/image", { method: "POST", body: form });
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

/* ------------------------------- submit -------------------------------- */
async function runCheck() {
  const text = articleText.value.trim();
  if (!text && !selectedFile) {
    note("Paste an article or attach a screenshot before checking.");
    (text ? articleText : dropzone).focus();
    return;
  }

  checkBtn.disabled = true;
  ctaSpinner.hidden = false;
  ctaLabel.textContent = "Checking\u2026";
  note("");
  showLoading(Boolean(selectedFile));

  try {
    const payload = selectedFile
      ? await verifyImage(selectedFile, text)
      : await verifyText(text);
    renderResult(payload, Boolean(selectedFile));
  } catch (error) {
    showError(error && error.message ? error.message : String(error));
  } finally {
    checkBtn.disabled = false;
    ctaSpinner.hidden = true;
    ctaLabel.textContent = "Check if News is Legit or Fake";
  }
}

/* --------------------------- backend status ---------------------------- */
async function refreshStatus() {
  const chip = $("statusChip");
  const text = $("statusText");
  try {
    const response = await fetch(API_BASE + "/health", { cache: "no-store" });
    const health = await response.json();
    const ready = health.status === "ok";
    chip.className = "topbar-status " + (ready ? "is-ok" : "is-warn");
    text.textContent = ready
      ? "System ready \u00b7 Live web search"
      : "System degraded";
  } catch (_) {
    $("statusChip").className = "topbar-status is-bad";
    $("statusText").textContent = "System unreachable";
  }
}

/* -------------------------------- wiring ------------------------------- */
articleText.addEventListener("input", updateWordCount);

$("clearBtn").addEventListener("click", () => {
  articleText.value = "";
  updateWordCount();
  note("");
});

$("pasteBtn").addEventListener("click", async () => {
  try {
    const clip = await navigator.clipboard.readText();
    if (clip) {
      articleText.value = clip;
      updateWordCount();
      articleText.focus();
    }
  } catch (_) {
    articleText.focus();
    note("Press Ctrl + V to paste (clipboard access was blocked by the browser).");
  }
});

document.querySelectorAll(".sample-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    const sample = SAMPLES[pill.dataset.sample];
    if (!sample) return;
    articleText.value = sample;
    updateWordCount();
    note("");
    articleText.focus();
  });
});

checkBtn.addEventListener("click", runCheck);
articleText.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runCheck();
});

updateWordCount();
refreshStatus();

