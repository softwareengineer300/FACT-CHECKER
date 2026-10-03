"""
backend.py
==========
Grounded fact-checking engine for text claims and images.

Stack
-----
* FastAPI + Uvicorn
* Google GenAI SDK (``google-genai``) driving a current Gemini Flash model
  (``gemini-3.6-flash`` by default, with automatic model fallbacks)
* Native Google Search grounding, so verdicts are backed by real-time web
  results and the response carries genuine, resolvable source URLs.

Endpoints
---------
GET  /health          Service + configuration status (never returns secrets).
GET  /config          Non-secret runtime configuration.
POST /verify/text     Fact-check a claim / news article   -> {"text": "..."}
POST /verify/image    Forensic analysis of an image       (multipart/form-data)
POST /analyze-text    Alias of /verify/text   (backwards compatible)
POST /analyze-image   Alias of /verify/image  (backwards compatible)

Every verification endpoint returns the same strict schema:

    {
      "verdict": "TRUE" | "FALSE" | "PARTIALLY_TRUE" | "UNVERIFIED",
      "truth_percentage": 0-100,
      "analysis": "...",
      "details": "...",
      "sources": [{"title": "...", "url": "..."}]
    }

Run
---
    uvicorn backend:app --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The merged single-page UI lives here and is served by this same app, so the
# frontend and the API share one process, one origin and one port.
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()  # also honour a .env living in the current working directory

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("factcheck.backend")

# --------------------------------------------------------------------------- #
# Optional SDK import.
# Importing defensively keeps the API bootable so /health can *explain* a
# broken environment instead of the process dying with a traceback.
# --------------------------------------------------------------------------- #
try:
    from google import genai
    from google.genai import types as genai_types

    GENAI_AVAILABLE = True
    GENAI_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - depends on the host environment
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    GENAI_AVAILABLE = False
    GENAI_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Ordered fallbacks used only when the configured model is retired/unavailable
# (Google returns 404 NOT_FOUND for models closed to new API keys) or when a
# per-model free-tier quota is exhausted.
MODEL_FALLBACKS = [
    name.strip()
    for name in os.getenv(
        "GEMINI_MODEL_FALLBACKS",
        "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,"
        "gemini-flash-lite-latest,gemini-3.8-flash,gemini-flash-latest",
    ).split(",")
    if name.strip()
]
PLACEHOLDER_KEYS = {"", "your_google_api_key_here", "changeme", "none", "null"}
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10 MiB
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "20000"))
GEMINI_ATTEMPTS = int(os.getenv("GEMINI_ATTEMPTS", "3"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

# --------------------------------------------------------------------------- #
# SQLite cache configuration
# --------------------------------------------------------------------------- #
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", os.path.join(BASE_DIR, "cache.db"))


def _init_cache() -> None:
    """Create the ``verifications`` table and migrate older databases."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verifications (
                    hash_key TEXT PRIMARY KEY,
                    query_type TEXT,
                    query_text TEXT,
                    response_json TEXT,
                    hit_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Safe migrations for databases created before the radar existed.
            existing = {
                row[1]
                for row in conn.execute("PRAGMA table_info(verifications)").fetchall()
            }
            if "query_text" not in existing:
                conn.execute("ALTER TABLE verifications ADD COLUMN query_text TEXT")
            if "hit_count" not in existing:
                conn.execute(
                    "ALTER TABLE verifications ADD COLUMN hit_count INTEGER DEFAULT 1"
                )
            conn.execute(
                "UPDATE verifications SET hit_count = 1 WHERE hit_count IS NULL"
            )
            conn.commit()
        log.info("SQLite cache ready at %s", CACHE_DB_PATH)
    except sqlite3.Error as exc:  # never let caching break the service
        log.warning("Could not initialise the SQLite cache: %s", exc)


def _text_hash(text: str) -> str:
    """Stable SHA-256 key for a text claim."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _image_hash(image_bytes: bytes, claim: Optional[str] = None) -> str:
    """Stable SHA-256 key for an image plus its optional caption."""
    return hashlib.sha256(
        image_bytes + (claim or "").encode("utf-8")
    ).hexdigest()


def _cache_get(hash_key: str) -> Optional[Dict[str, Any]]:
    """Return the cached payload for ``hash_key``, or ``None`` on a miss."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            row = conn.execute(
                "SELECT response_json FROM verifications WHERE hash_key = ?",
                (hash_key,),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE verifications SET hit_count = hit_count + 1 "
                    "WHERE hash_key = ?",
                    (hash_key,),
                )
                conn.commit()
    except sqlite3.Error as exc:
        log.warning("Cache lookup failed for %s: %s", hash_key[:12], exc)
        return None
    if not row:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError) as exc:
        log.warning("Corrupt cache row for %s: %s", hash_key[:12], exc)
        return None
    if isinstance(payload, dict):
        payload["cached"] = True
        return payload
    return None


def _cache_put(
    hash_key: str,
    query_type: str,
    payload: Dict[str, Any],
    query_text: str = "",
) -> None:
    """Persist a freshly computed verdict, tagged with ``cached: false``.

    ``query_text`` records the original claim so the trending radar can report
    what people are actually checking.
    """
    body = dict(payload)
    body["cached"] = False
    snippet = " ".join((query_text or "").split())[:500]
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO verifications
                    (hash_key, query_type, query_text, response_json,
                     hit_count, created_at)
                VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(hash_key) DO UPDATE SET
                    query_type = excluded.query_type,
                    query_text = excluded.query_text,
                    response_json = excluded.response_json
                """,
                (
                    hash_key,
                    query_type,
                    snippet,
                    json.dumps(body, ensure_ascii=False),
                ),
            )
            conn.commit()
    except sqlite3.Error as exc:
        log.warning("Could not write cache row %s: %s", hash_key[:12], exc)

# Magic-number sniffing used to validate the *actual* payload, not just the
# client-supplied Content-Type header.
_MAGIC_SIGNATURES = (
    ("image/jpeg", 0, b"\xff\xd8\xff"),
    ("image/png", 0, b"\x89PNG\r\n\x1a\n"),
    ("image/webp", 8, b"WEBP"),
)

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class GeminiUnavailable(RuntimeError):
    """The service cannot call Gemini (missing SDK, missing/invalid API key)."""


class GeminiFailure(RuntimeError):
    """Gemini was reachable but the call ultimately failed."""


class ModelOutputError(RuntimeError):
    """Gemini answered, but the payload was not valid against our schema."""


# --------------------------------------------------------------------------- #
# Strict response schema
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    """The only verdicts the system is allowed to emit."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    UNVERIFIED = "UNVERIFIED"


class SourceItem(BaseModel):
    """A single, resolvable citation returned alongside a verdict."""

    title: str = Field(..., description="Short human-readable label for the source.")
    url: str = Field(..., description="Absolute URL of the source.")


class FactCheckResult(BaseModel):
    """The strict JSON contract shared by both verification endpoints."""

    verdict: Verdict = Field(..., description="TRUE, FALSE, PARTIALLY_TRUE or UNVERIFIED.")
    truth_percentage: float = Field(
        ..., ge=0, le=100, description="Estimated factual accuracy, 0-100."
    )
    analysis: str = Field(..., description="Core reasoning and claim breakdown.")
    details: str = Field(
        ...,
        description=(
            "Verified additional context when TRUE; the factual correction when "
            "FALSE or PARTIALLY_TRUE."
        ),
    )
    sources: List[SourceItem] = Field(
        default_factory=list, description="Sources backing the verdict."
    )
    cached: bool = Field(
        default=False, description="True when served instantly from the local cache."
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value: Any) -> Any:
        """Accept the enum, the exact strings, and common model paraphrases."""
        if isinstance(value, Verdict):
            return value
        if value is None:
            raise ValueError("verdict is required")
        key = re.sub(r"[^A-Z]", "", str(value).upper())
        aliases = {
            "TRUE": Verdict.TRUE,
            "REAL": Verdict.TRUE,
            "GENUINE": Verdict.TRUE,
            "AUTHENTIC": Verdict.TRUE,
            "CORRECT": Verdict.TRUE,
            "FALSE": Verdict.FALSE,
            "FAKE": Verdict.FALSE,
            "UNTRUE": Verdict.FALSE,
            "FABRICATED": Verdict.FALSE,
            "HOAX": Verdict.FALSE,
            "PARTIALLYTRUE": Verdict.PARTIALLY_TRUE,
            "PARTLYTRUE": Verdict.PARTIALLY_TRUE,
            "HALFTRUE": Verdict.PARTIALLY_TRUE,
            "MOSTLYTRUE": Verdict.PARTIALLY_TRUE,
            "MIXED": Verdict.PARTIALLY_TRUE,
            "MISLEADING": Verdict.PARTIALLY_TRUE,
            "CONTEXTMISSING": Verdict.PARTIALLY_TRUE,
            "UNVERIFIED": Verdict.UNVERIFIED,
            "UNVERIFIABLE": Verdict.UNVERIFIED,
            "UNKNOWN": Verdict.UNVERIFIED,
            "INSUFFICIENT": Verdict.UNVERIFIED,
            "INSUFFICIENTEVIDENCE": Verdict.UNVERIFIED,
            "NOEVIDENCE": Verdict.UNVERIFIED,
            "NOTVERIFIED": Verdict.UNVERIFIED,
            "INCONCLUSIVE": Verdict.UNVERIFIED,
            "CANNOTVERIFY": Verdict.UNVERIFIED,
            "CANNOTDETERMINE": Verdict.UNVERIFIED,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"Unrecognised verdict: {value!r}")

    @field_validator("truth_percentage", mode="before")
    @classmethod
    def _normalise_percentage(cls, value: Any) -> Any:
        """Coerce "85", "85%", 85.5 or None into a clamped float."""
        if isinstance(value, str):
            cleaned = re.sub(r"[^0-9.]", "", value)
            value = cleaned or 0
        try:
            percentage = float(value)
        except (TypeError, ValueError):
            percentage = 0.0
        return max(0.0, min(100.0, percentage))

    @field_validator("analysis", "details", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> Any:
        """Models occasionally answer with a list/dict; flatten it to text."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            return "\n".join(
                json.dumps(item, ensure_ascii=False)
                if isinstance(item, (dict, list))
                else str(item).strip()
                for item in value
            )
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    @field_validator("sources", mode="before")
    @classmethod
    def _normalise_sources(cls, value: Any) -> Any:
        """Accept [{"title","url"}], bare URL strings, or uri/link key variants."""
        if not value:
            return []
        if isinstance(value, (str, dict)):
            value = [value]
        normalised: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, str):
                url = item.strip()
                if url:
                    normalised.append({"title": _domain_of(url) or url, "url": url})
                continue
            if not isinstance(item, dict):
                continue
            url = str(
                item.get("url")
                or item.get("uri")
                or item.get("link")
                or item.get("source_url")
                or ""
            ).strip()
            if not url:
                continue
            title = str(
                item.get("title") or item.get("name") or item.get("source") or ""
            ).strip()
            normalised.append({"title": title or _domain_of(url) or url, "url": url})
        return normalised


class TextVerificationRequest(BaseModel):
    """Body of ``POST /verify/text``."""

    text: str = Field(..., description="The claim, headline or article to verify.")

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, value: Any) -> Any:
        """Tolerate ``claim`` / ``query`` / ``article`` as the payload key."""
        if isinstance(value, dict) and not value.get("text"):
            for alias in ("claim", "query", "article", "content", "url", "input"):
                if value.get(alias):
                    return {"text": value[alias]}
        return value

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("text is required")
        text = str(value).strip()
        if len(text) < 3:
            raise ValueError("text must contain at least 3 characters")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(
                f"text is too long ({len(text)} chars); limit is {MAX_TEXT_CHARS}"
            )
        return text


class HealthResponse(BaseModel):
    """Machine-readable readiness report for ``GET /health``."""

    status: str
    model: str
    sdk_installed: bool
    api_key_configured: bool
    search_grounding: bool
    detail: str


# --------------------------------------------------------------------------- #
# Endpoint callbacks + hooks
# --------------------------------------------------------------------------- #
_TRANSIENT_MARKERS = (
    "500", "502", "503", "504",
    "INTERNAL", "UNAVAILABLE", "DEADLINE", "OVERLOADED",
)
# NOTE: 429/RESOURCE_EXHAUSTED is deliberately absent. The SDK already retries
# 429 internally, and free-tier quota exhaustion is not a transient hiccup --
# retrying only burns more of a per-model daily allowance.
_FATAL_MARKERS = (
    "api key not valid", "api_key_invalid", "unauthenticated", "permission", "401", "403",
)


def _is_transient_error(error: Exception) -> bool:
    """True when a Gemini failure looks like a retryable transport/model hiccup."""
    text = f"{type(error).__name__}: {error}".upper()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _is_fatal_error(error: Exception) -> bool:
    """True for auth/permission failures that retrying can never fix."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _FATAL_MARKERS)


def _reject_function_calls(response: Any) -> None:
    """Guard: grounding tools must answer in prose, never via a function call.

    ``GenerateContentConfig`` in google-genai >= 2.x exposes no ``on_tool_call``
    hook, so the check happens on the finished response instead. Declaring only
    ``google_search`` / ``url_context`` means a function call is always a model
    misfire, and re-asking is safer than parsing a non-answer.
    """
    calls = getattr(response, "function_calls", None) or []
    if calls:
        names = ", ".join(getattr(fc, "name", "?") for fc in calls)
        raise ModelOutputError(
            f"Gemini attempted an unexpected function call ({names}) instead of "
            "answering with grounded prose. Please retry."
        )


# --------------------------------------------------------------------------- #
# Gemini client management
# --------------------------------------------------------------------------- #
class GeminiManager:
    """Owns the lazy, thread-safe, self-healing Gemini client.

    The client is created on first use. If the API key changes on disk (e.g. the
    user fills in ``.env`` after the server started) it is transparently rebuilt,
    so no restart is required.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._client: Optional[Any] = None
        self._client_key: Optional[str] = None
        self._http_options: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    # -- configuration ----------------------------------------------------- #
    @staticmethod
    def api_key() -> str:
        """Return a usable API key, preferring the environment over ``.env``."""
        candidates = (
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_GENAI_API_KEY"),
        )
        for candidate in candidates:
            if candidate and candidate.strip().lower() not in PLACEHOLDER_KEYS:
                return candidate.strip()
        return ""

    @classmethod
    def api_key_configured(cls) -> bool:
        return bool(cls.api_key())

    # -- http options ------------------------------------------------------ #
    def http_options(self) -> Dict[str, Any]:
        """Build ``http_options``: unified ``v1beta`` endpoint + retry policy."""
        if self._http_options is None:
            options: Dict[str, Any] = {"api_version": "v1beta", "timeout": 120_000}
            retry_cls = getattr(genai_types, "HttpRetryOptions", None)
            if retry_cls is not None:
                options["retry_options"] = retry_cls(
                    attempts=max(1, GEMINI_ATTEMPTS),
                    initial_delay=1.0,
                    max_delay=12.0,
                    exp_base=2.0,
                    jitter=1.0,
                    # NOTE: 429 is deliberately excluded. Quota exhaustion is not a
                    # transient hiccup, and the SDK's own retry loop would add ~10s
                    # of dead time per model before our loop could move on to the
                    # next candidate. Letting the 429 surface immediately makes the
                    # per-model fallback (see ``_is_quota_exhausted``) fast.
                    http_status_codes=[500, 502, 503, 504],
                )
            self._http_options = options
        return self._http_options

    # -- client ------------------------------------------------------------ #
    def client(self) -> Any:
        """Return a ready client or raise :class:`GeminiUnavailable`."""
        if not GENAI_AVAILABLE:
            raise GeminiUnavailable(
                "The analysis engine is not available"
                f"{f' ({GENAI_IMPORT_ERROR})' if GENAI_IMPORT_ERROR else ''}."
                " Contact the administrator."
            )
        key = self.api_key()
        if not key:
            raise GeminiUnavailable(
                "The analysis engine is not configured. Contact the administrator."
            )
        with self._lock:
            if self._client is None or self._client_key != key:
                try:
                    self._client = genai.Client(
                        api_key=key, http_options=self.http_options()
                    )
                    self._client_key = key
                    log.info("Gemini client initialised (model=%s)", self.model_name)
                except Exception as exc:
                    raise GeminiUnavailable(
                        f"Could not initialise the Gemini client: {exc}"
                    ) from exc
            return self._client

    def status(self) -> Dict[str, Any]:
        """Secret-free configuration snapshot used by ``/health``."""
        return {
            "sdk_installed": GENAI_AVAILABLE,
            "api_key_configured": self.api_key_configured(),
            "model": self.model_name,
        }


MANAGER = GeminiManager()


def _domain_of(url: str) -> str:
    """Extract a bare hostname from a URL for use as a fallback title."""
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://([^/?#]+)", url or "")
    if match:
        host = match.group(1)
        return host[4:] if host.lower().startswith("www.") else host
    return ""


def _dedupe_sources(sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate URLs while keeping citation order stable."""
    seen: set = set()
    unique: List[Dict[str, str]] = []
    for item in sources:
        url = (item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append({"title": item.get("title") or _domain_of(url) or url, "url": url})
    return unique


# --------------------------------------------------------------------------- #
# Model instructions
# --------------------------------------------------------------------------- #
JSON_SCHEMA_HINT = """{
  "verdict": "TRUE" | "FALSE" | "PARTIALLY_TRUE" | "UNVERIFIED",
  "truth_percentage": <number from 0 to 100>,
  "analysis": "<string>",
  "details": "<string>",
  "sources": [{"title": "<string>", "url": "<string>"}]
}"""

LANGUAGE_AND_SOURCE_RULES = """
LANGUAGE AND REGIONAL SOURCING
- Detect the language of the claim. If the claim is written in Marathi, Hindi, or a regional Indian context, generate every value of "analysis", "details", and all explanations entirely in standard Marathi (Marathi Devanagari script). Keep the strict JSON keys (verdict, truth_percentage, analysis, details, sources) in English.
- For claims involving Indian policies, viral social media messages, or regional news, prioritise reputable fact-check desks such as PIB Fact Check, Boom Live, and Vishvas News.
"""


_OUTPUT_RULES = f"""
OUTPUT CONTRACT
- Reply with exactly ONE JSON object and nothing else.
- No markdown fences, no preamble, no trailing commentary, no keys beyond the five below.
- Every key is mandatory. "sources" is always present (use [] if genuinely none).

{JSON_SCHEMA_HINT}
"""

TEXT_SYSTEM_INSTRUCTION = f"""You are "Veritas", an impartial, evidence-first fact-checking analyst \
with live access to Google Search. You are given a claim, headline, article, or URL. You must \
determine whether it is TRUE, FALSE, or PARTIALLY_TRUE and justify it with sources a reader can open.

METHOD (follow in order)
1. Restate the input as a numbered list of atomic, independently checkable factual assertions.
   Strip rhetoric, opinion, and speculation; keep names, numbers, dates, places, and events.
2. Search for authoritative coverage of every atomic assertion. Prefer, in order: primary sources \
(official statements, court/government records, company filings, peer-reviewed work, the original \
article or wire service) and then high-quality reporting from established outlets. Treat social \
posts, anonymous blogs, aggregators, and AI-generated content as low-weight evidence.
3. Establish the timeline. Determine what was actually said or reported, when, and by whom, and \
whether the input is a distortion, a re-dating, or a recycled older story.
4. Decide the verdict and the truth percentage.

VERDICT RULES
- TRUE: every material assertion is corroborated by reliable sources, and the framing is accurate.
- FALSE: the core assertion is contradicted by reliable sources, or refers to an event that never \
happened, or is a fabrication, hoax, or manipulated artefact.
- PARTIALLY_TRUE: a real event or statistic is presented with the wrong date, place, actor, \
magnitude, or causality; or the input mixes accurate and inaccurate assertions; or true facts are \
paired with misleading framing, missing context, or an unverified attribution. Use this verdict \
whenever the input is partly accurate and partly wrong - do not round it to TRUE or FALSE.

FIELD-BY-FIELD REQUIREMENTS
- truth_percentage: a number from 0 to 100 expressing how factually accurate the input is exactly as \
stated. Guidance: fully accurate 95-100; accurate but minor imprecision 80-94; mixed accuracy \
45-79; mostly wrong 10-44; fabricated 0-9. Be decisive and avoid always clustering near 50.
- analysis: your reasoning, as plain prose in 2-5 short paragraphs. Include (a) the numbered claim \
breakdown, (b) what the searches actually showed, (c) where sources agree or conflict and how you \
weighed them, and (d) the single decisive reason for the verdict.
- details: additional value for the reader.
    * If the verdict is TRUE, give the verified context they would not get from the bare claim: \
background, chronology, the underlying data, the primary source, and the significance of the story.
    * If the verdict is FALSE or PARTIALLY_TRUE, give the factual correction: what is actually true, \
stated positively and concretely (real names, dates, numbers, places), plus the likely origin of the \
false version when the sources reveal it.
- sources: every URL you actually relied on, each with a short descriptive title (outlet + subject). \
Use only URLs that appeared in your search results. Never construct, guess, or shorten a URL.

HANDLING MISINFORMATION - "HOW THIS SPREAD ONLINE"
When the verdict is FALSE or PARTIALLY_TRUE you MUST append a final paragraph to "analysis" that \
begins with the exact label "How this spread online:" and describes the observable propagation of \
the claim: the earliest appearance you can find, the platforms, outlets, accounts, hashtags, or \
communities that carried it, any notable amplification or spike, whether fact-checkers have already \
debunked it, and whether platforms have labelled or removed it. Base this strictly on what your \
search results actually show. If the available results do not reveal the propagation path, write \
"The available search results do not establish a clear propagation path." Do not speculate.

INTEGRITY RULES
- Never invent facts, quotations, statistics, dates, names, or URLs. If it is not in the search \
results, do not assert it.
- Never present an unverified claim as established fact.
- If search results are thin, contradictory, or unavailable, say so explicitly in "analysis", keep \
truth_percentage conservative, and put at most a small number of well-supported URLs in "sources".
- Do not soften a FALSE verdict to be polite, and do not inflate a TRUE verdict to be agreeable. \
Your only loyalty is to the evidence.

{LANGUAGE_AND_SOURCE_RULES}
{_OUTPUT_RULES}"""

IMAGE_SYSTEM_INSTRUCTION = f"""You are "Veritas Forensics", an impartial image-authenticity and \
misinformation analyst with live access to Google Search. You receive an image, optionally with a \
caption or claim about where it came from and what it shows. Determine whether the image and its \
alleged context are TRUE, FALSE, or PARTIALLY_TRUE, and justify it with sources a reader can open.

METHOD (follow in order)
1. Describe the image factually: subject matter, people, objects, setting, any visible text, logos, \
signage, registration plates, uniforms, and the apparent time of day or season.
2. MANIPULATION FORENSICS. Examine, and comment only on what you can actually observe:
   - Generative or editing artefacts: unnatural skin or hair texture, melting or fused edges, \
malformed hands, fingers, teeth, or ears, incoherent background geometry, duplicated or warped \
objects, impossible reflections, garbled or nonsensical text, watermark or logo corruption.
   - Composite or splice indicators: mismatched lighting direction, colour temperature, shadow \
direction, or perspective; hard or blurred edges around an inserted subject; inconsistent grain or \
noise between regions; halo or clipping artefacts along a cut-out boundary.
   - Metadata and provenance indicators: embedded EXIF data if present, camera make and model, \
software strings, creation timestamps, GPS coordinates, and any editor or AI-generator signature. \
Explicitly state when no metadata is available.
   - Compression forensics: double-JPEG artefacts, error-level or noise inconsistency between \
regions, resampling traces, or cloning.
   - Screenshot and re-photograph indicators: capture UI elements, moire patterns, keystone \
distortion, or screen glare.
3. CONTEXT VERIFICATION. Use Google Search to check the alleged context: reverse-image leads, the \
event, place, and date claimed, the people and uniforms shown, and whether the same image has been \
published elsewhere with a different or older story. A genuine, unedited photograph presented with \
the wrong date, place, or event is PARTIALLY_TRUE or FALSE, not TRUE.
4. Weigh everything and decide.

VERDICT RULES
- TRUE: the image shows no observable manipulation AND its alleged context is corroborated by \
reliable sources.
- FALSE: the image is AI-generated, synthesised, deepfaked, or materially edited; or it depicts \
something that did not happen; or its alleged context is fabricated.
- PARTIALLY_TRUE: the image is genuine but re-captioned, re-dated, or recycled from another event; \
or it is a real photo that has been cosmetically altered in a way that changes its meaning; or \
authenticity cannot be confirmed while part of the context checks out.

FIELD-BY-FIELD REQUIREMENTS
- truth_percentage: 0-100, expressing the overall truthfulness of the image AND its alleged context \
taken together. A pristine photo attached to a false caption is not a high percentage.
- analysis: your reasoning in 2-5 short paragraphs, covering (a) what the image shows, (b) the \
manipulation findings and their significance, (c) what the searches showed about the alleged \
context, and (d) the decisive reason for the verdict. Be explicit about uncertainty: state clearly \
when you cannot determine whether an image is synthetic rather than guessing.
- details: if TRUE, the verified background of the image - the real event, place, date, photographer, \
or original publication. If FALSE or PARTIALLY_TRUE, explain what the image really is, where it \
actually comes from, how the context was altered, and any established debunking by fact-checkers or \
news agencies.
- sources: every URL you actually relied on, each with a short descriptive title. Use only URLs from \
your search results; never construct, guess, or shorten a URL.

HANDLING MISINFORMATION - "HOW THIS SPREAD ONLINE"
When the verdict is FALSE or PARTIALLY_TRUE you MUST append a final paragraph to "analysis" beginning \
with the exact label "How this spread online:" describing the observable propagation of the image: \
where it first appeared, the platforms, pages, accounts, or hashtags that shared it, any viral spike, \
and whether fact-checkers or platforms have already addressed it. Base this strictly on your search \
results. If they do not reveal a propagation path, write "The available search results do not \
establish a clear propagation path." Do not speculate.

INTEGRITY RULES
- Never claim certainty about synthetic origin that the visual evidence does not support. Say what \
you can see and label it as an indicator, not proof.
- Never invent metadata, EXIF values, photographers, events, or URLs.
- If the image is unreadable, corrupt, or too low-resolution to assess, say so plainly and return a \
low-confidence verdict with an honest explanation rather than a confident guess.

{LANGUAGE_AND_SOURCE_RULES}
{_OUTPUT_RULES}"""

# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #
def _build_tools() -> List[Any]:
    """Native grounding tools: Google Search, plus URL Context when available."""
    if not GENAI_AVAILABLE:
        return []
    tools: List[Any] = []
    search_cls = getattr(genai_types, "GoogleSearch", None)
    tool_cls = getattr(genai_types, "Tool", None)
    if tool_cls is None:
        return [{"google_search": {}}]
    if search_cls is not None:
        try:
            tools.append(tool_cls(google_search=search_cls()))
        except Exception as exc:  # pragma: no cover - version specific
            log.warning("Could not build GoogleSearch tool (%s); using dict form", exc)
            tools.append(tool_cls(google_search={}))
    else:
        tools.append(tool_cls(google_search={}))
    url_context_cls = getattr(genai_types, "UrlContext", None)
    if url_context_cls is not None:
        try:
            tools.append(tool_cls(url_context=url_context_cls()))
        except Exception as exc:  # pragma: no cover - version specific
            log.debug("UrlContext tool unavailable: %s", exc)
    return tools


def _build_schema() -> Optional[Any]:
    """Strict structured-output schema, mirroring :class:`FactCheckResult`."""
    if not GENAI_AVAILABLE:
        return None
    schema_cls = getattr(genai_types, "Schema", None)
    if schema_cls is None:
        return None
    verdict_cls = getattr(genai_types, "Type", None)
    verdict_field: Dict[str, Any] = {
        "type": "STRING",
        "enum": ["TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIED"],
        "description": "TRUE, FALSE, PARTIALLY_TRUE or UNVERIFIED.",
    }
    if verdict_cls is not None and hasattr(verdict_cls, "STRING"):
        verdict_field["type"] = verdict_cls.STRING
    try:
        return schema_cls(
            type=verdict_field["type"],
            properties={
                "verdict": schema_cls(**verdict_field),
                "truth_percentage": schema_cls(
                    type="NUMBER",
                    description="Estimated factual accuracy from 0 to 100.",
                ),
                "analysis": schema_cls(
                    type="STRING",
                    description=(
                        "Reasoning, claim breakdown, evidence weighing, and for FALSE or "
                        "PARTIALLY_TRUE a final 'How this spread online:' paragraph."
                    ),
                ),
                "details": schema_cls(
                    type="STRING",
                    description=(
                        "Verified additional context when TRUE; the factual correction "
                        "when FALSE or PARTIALLY_TRUE."
                    ),
                ),
                "sources": schema_cls(
                    type="ARRAY",
                    description="Sources backing the verdict.",
                    items=schema_cls(
                        type="OBJECT",
                        properties={
                            "title": schema_cls(type="STRING"),
                            "url": schema_cls(type="STRING"),
                        },
                        required=["title", "url"],
                    ),
                ),
            },
            required=[
                "verdict",
                "truth_percentage",
                "analysis",
                "details",
                "sources",
            ],
        )
    except Exception as exc:  # pragma: no cover - version specific
        log.warning("Structured-output schema unavailable (%s); prompt-enforced JSON", exc)
        return None


def _build_config(
    system_instruction: str, schema: Optional[Any], include_tools: bool = True
) -> Any:
    """Assemble ``GenerateContentConfig`` for one verification call.

    ``response_schema`` is attached **only** when search grounding is not in use.
    The Gemini API rejects ``response_schema`` combined with the ``google_search``
    tool with ``400 INVALID_ARGUMENT``, so attaching it on the grounded path would
    burn a request (and quota) on a guaranteed failure. When grounding is enabled
    the JSON shape is enforced by the system prompt instead, and the reply is
    parsed by :func:`_extract_json`.
    """
    kwargs: Dict[str, Any] = {
        "system_instruction": system_instruction,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    tools: List[Any] = _build_tools() if include_tools else []
    if tools:
        kwargs["tools"] = tools
    if schema is not None and not tools:
        kwargs["response_schema"] = schema
    return genai_types.GenerateContentConfig(**kwargs)


def _model_candidates() -> List[str]:
    """Configured model first, then configured fallbacks (order preserved)."""
    ordered: List[str] = []
    for name in [MANAGER.model_name, *MODEL_FALLBACKS]:
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def _is_model_missing(error: Exception) -> bool:
    """True when the API reports the model as absent or retired (404)."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in ("404", "not_found", "no longer available", "not found")
    )


def _is_quota_exhausted(error: Exception) -> bool:
    """True when this model's own daily/rate quota is spent (429).

    Free-tier quota is metered **per model**, so moving to the next candidate is
    worthwhile: a rejected request consumes no quota and a sibling model may
    still have its allowance intact.
    """
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in ("429", "resource_exhausted", "quota exceeded", "rate limit")
    )


GROUNDED_SUFFIX = (
    "\n\nTOOL USAGE: You have live Google Search grounding. Search the web before "
    "answering. Never emit a function call; always reply with the JSON object."
)

UNGROUNDED_SUFFIX = (
    "\n\nTOOL USAGE: Web search is unavailable for this request. Do NOT attempt to "
    "call any function or tool and do NOT claim to have searched the web. Answer "
    "directly from your own knowledge, say so where relevant, and always reply with "
    "the JSON object requested above."
)


def _generate(system_instruction: str, contents: List[Any], schema: Optional[Any]) -> Any:
    """Call Gemini, degrading gracefully across schemas, models and tools.

    Two tool passes are attempted:

    1. **Grounded** - with ``google_search`` attached, so the verdict carries real
       source URLs. This is the normal path.
    2. **Ungrounded** - only reached when *every* candidate model reports the
       search-grounding quota as exhausted (``429``). Grounding is metered
       separately from plain generation on the free tier, so the same model often
       still answers without tools. The verdict is then based on the model's own
       knowledge and may carry no sources, which the UI states explicitly.

    Within a pass, ``response_schema`` is only used when no tools are attached
    (the API rejects that combination), a retired model (404) advances to the next
    candidate, a per-model quota exhaustion (429) advances too, and transient
    5xx/overload errors get one extra try.
    """
    client = MANAGER.client()
    last_error: Optional[Exception] = None
    grounding_blocked = False
    tool_passes = [True, False] if _build_tools() else [False]

    for use_tools in tool_passes:
        if not use_tools and not grounding_blocked:
            break  # grounding worked; never silently downgrade
        if schema is not None and not use_tools:
            schema_passes: List[Optional[Any]] = [schema, None]
        else:
            schema_passes = [None]

        for model_name in _model_candidates():
            next_model = False
            for active_schema in schema_passes:
                if next_model:
                    break
                for transient_pass in (False, True):
                    try:
                        active_instruction = system_instruction + (
                            GROUNDED_SUFFIX if use_tools else UNGROUNDED_SUFFIX
                        )
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=_build_config(
                                active_instruction, active_schema, include_tools=use_tools
                            ),
                        )
                        _reject_function_calls(response)
                        if model_name != MANAGER.model_name:
                            log.warning("Answered using fallback model %s.", model_name)
                        if active_schema is None and not use_tools:
                            log.info("Used prompt-enforced JSON (no response_schema).")
                        if not use_tools:
                            log.warning(
                                "Search grounding is out of quota; answered WITHOUT live "
                                "web search, so the verdict may carry no sources."
                            )
                        return response
                    except (GeminiUnavailable, ModelOutputError):
                        raise
                    except Exception as exc:
                        last_error = exc
                        if _is_fatal_error(exc):
                            raise GeminiFailure(_friendly_error(exc)) from exc
                        if _is_model_missing(exc):
                            log.warning(
                                "Model %s unavailable (%s); trying next candidate.",
                                model_name,
                                exc,
                            )
                            next_model = True
                            break
                        if _is_quota_exhausted(exc):
                            if use_tools:
                                grounding_blocked = True
                                log.warning(
                                    "Model %s has no search-grounding quota left; "
                                    "trying next candidate.",
                                    model_name,
                                )
                            else:
                                log.warning(
                                    "Model %s has no quota left; trying next candidate.",
                                    model_name,
                                )
                            next_model = True
                            break
                        if _is_transient_error(exc) and not transient_pass:
                            log.warning("Transient Gemini error (%s); retrying.", exc)
                            continue
                        break
    raise GeminiFailure(_friendly_error(last_error))


def _friendly_error(error: Optional[Exception]) -> str:
    """Translate SDK/transport failures into an actionable message."""
    if error is None:
        return "The analysis request failed for an unknown reason."
    text = f"{type(error).__name__}: {error}"
    lowered = text.lower()

    if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:
        hint = ""
        match = re.search(r"retry in ([0-9a-z\. ]+?)(?:\.|,|$)", lowered)
        if match:
            hint = f" Quota resets in about {match.group(1).strip()}."
        return (
            "Live web search quota is exhausted (429). The free tier limits how many "
            "requests can use live search per day. Wait for the limit to reset or "
            f"contact the administrator.{hint} Details: {text}"
        )
    if "permission" in lowered or "403" in lowered:
        return f"Access denied (403). Verify the service is enabled. Details: {text}"
    if (
        "api key not valid" in lowered
        or "api_key_invalid" in lowered
        or "unauthenticated" in lowered
    ):
        return f"Authentication failed. Contact the administrator. Details: {text}"
    if "not_found" in lowered or "no longer available" in lowered:
        return f"The analysis engine is temporarily unavailable. Contact the administrator. Details: {text}"
    if "invalid_argument" in lowered or "400" in lowered:
        return f"The request was rejected (400). Details: {text}"
    if "unavailable" in lowered or "503" in lowered or "overloaded" in lowered:
        return f"The service is temporarily overloaded (503). Retry shortly. Details: {text}"
    if "deadline" in lowered or "timeout" in lowered or "timed out" in lowered:
        return f"The request timed out. Retry, or use a smaller image. Details: {text}"
    if "safety" in lowered or "blocked" in lowered or "prohibited" in lowered:
        return f"The request was blocked for safety reasons. Details: {text}"
    return f"The analysis request failed: {text}"


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> Dict[str, Any]:
    """Pull the first JSON object out of the model's reply."""
    if not raw or not raw.strip():
        raise ModelOutputError("Gemini returned an empty response.")
    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ModelOutputError(
                f"The analysis engine did not return JSON. Raw reply started with: {raw[:200]!r}"
            ) from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelOutputError(
                f"The analysis engine returned malformed JSON ({exc}). Raw reply started with: {raw[:200]!r}"
            ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("The analysis engine returned JSON that is not an object.")
    return payload


def _coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise the raw payload so Pydantic validation cannot trip on trivia."""
    data = dict(payload)
    # Some builds nest the answer under a wrapper key.
    for wrapper in ("result", "response", "fact_check", "factcheck", "data", "output"):
        inner = data.get(wrapper)
        if isinstance(inner, dict) and any(
            key in inner for key in ("verdict", "analysis", "truth_percentage")
        ):
            data = {**data, **inner}
            break
    data.setdefault("analysis", "")
    data.setdefault("details", "")
    data.setdefault("sources", [])
    if not data.get("verdict"):
        raise ModelOutputError("The analysis response is missing the 'verdict' field.")
    if data.get("truth_percentage") is None:
        data["truth_percentage"] = 0
    return data


def _has_grounding_metadata(response: Any) -> bool:
    """True when Gemini actually used Google Search for this response."""
    try:
        for candidate in getattr(response, "candidates", None) or []:
            if getattr(candidate, "grounding_metadata", None) is not None:
                return True
    except Exception:  # pragma: no cover - defensive
        return False
    return False


def _grounding_sources(response: Any) -> List[Dict[str, str]]:
    """Extract real citation URLs from the grounding metadata of a response."""
    collected: List[Dict[str, str]] = []
    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            if metadata is None:
                continue
            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web is None:
                    continue
                uri = (getattr(web, "uri", None) or "").strip()
                if not uri:
                    continue
                title = (getattr(web, "title", None) or "").strip()
                domain = (getattr(web, "domain", None) or "").strip()
                collected.append(
                    {"title": title or domain or _domain_of(uri) or uri, "url": uri}
                )
            for query in getattr(metadata, "web_search_queries", None) or []:
                log.debug("Grounding search query used: %s", query)
    except Exception as exc:  # never let citation parsing break a verdict
        log.warning("Could not parse grounding metadata: %s", exc)
    return _dedupe_sources(collected)


def _response_text(response: Any) -> str:
    """Best-effort text extraction that also surfaces blocked responses."""
    text = getattr(response, "text", None)
    if text and text.strip():
        return text
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            chunk = getattr(part, "text", None)
            if chunk and chunk.strip():
                return chunk
        finish = getattr(candidate, "finish_reason", None)
        if finish and str(finish).upper() not in ("STOP", "FINISH_REASON_STOP"):
            raise ModelOutputError(
                f"Gemini stopped before producing an answer (finish_reason={finish}). "
                "The input may have been blocked or the response truncated."
            )
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            raise ModelOutputError(
                f"Gemini refused the request (block_reason={block_reason})."
            )
    raise ModelOutputError("Gemini returned no usable text.")


def _parse_result(response: Any) -> FactCheckResult:
    """Validate Gemini's reply and merge in grounding-derived citations."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, FactCheckResult):
        result = parsed
    elif isinstance(parsed, BaseModel):
        result = FactCheckResult(**_coerce_payload(parsed.model_dump()))
    elif isinstance(parsed, dict):
        result = FactCheckResult(**_coerce_payload(parsed))
    else:
        result = FactCheckResult(**_coerce_payload(_extract_json(_response_text(response))))

    merged = _dedupe_sources(
        [item.model_dump() for item in result.sources] + _grounding_sources(response)
    )
    if len(merged) != len(result.sources):
        result.sources = [SourceItem(**item) for item in merged]

    # Be explicit when the free tier ran out of live search grounding, so a
    # knowledge-only verdict is never mistaken for a web-verified one.
    if not _has_grounding_metadata(response) and not result.sources:
        note = (
            "Note: this verdict was produced without live web search "
            "(the search quota is exhausted), so it reflects the "
            "model's own knowledge and no source URLs could be attached."
        )
        result.details = f"{result.details}\n\n{note}".strip()
    return result


# --------------------------------------------------------------------------- #
# Verification logic
# --------------------------------------------------------------------------- #
def _sniff_mime(raw: bytes) -> str:
    """Identify the real format from magic bytes (never trust the header alone)."""
    for mime, offset, signature in _MAGIC_SIGNATURES:
        if raw[offset : offset + len(signature)] == signature:
            return mime
    return ""


def _validate_image(raw: bytes, declared_mime: Optional[str]) -> str:
    """Validate size and true format; return the authoritative MIME type."""
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is {len(raw) / 1048576:.1f} MB; the limit is "
                f"{MAX_IMAGE_BYTES // 1048576} MB."
            ),
        )
    sniffed = _sniff_mime(raw)
    if not sniffed:
        raise HTTPException(
            status_code=415,
            detail="Unsupported or corrupt image. Upload a valid JPG, PNG or WEBP file.",
        )
    if declared_mime and declared_mime.split(";")[0].strip().lower() not in ALLOWED_IMAGE_TYPES:
        log.warning("Client declared %r but bytes are %s; trusting the bytes.", declared_mime, sniffed)
    return sniffed


def verify_text(text: str) -> FactCheckResult:
    """Fact-check a text claim/article with search grounding (cache-first)."""
    hash_key = _text_hash(text)
    hit = _cache_get(hash_key)
    if hit is not None:
        log.info("Cache HIT for text claim (%s)", hash_key[:12])
        return FactCheckResult(**_coerce_payload(hit))

    contents: List[Any] = [
        "CLAIM / ARTICLE TO FACT-CHECK\n"
        "========================================\n"
        f"{text}\n"
        "========================================\n\n"
        "Search the web, verify the material above, and return the required JSON object."
    ]
    response = _generate(TEXT_SYSTEM_INSTRUCTION, contents, _build_schema())
    result = _parse_result(response)
    _cache_put(hash_key, "text", result.model_dump(), query_text=text)
    result.cached = False
    return result


def verify_image(raw: bytes, mime: str, claim: Optional[str] = None) -> FactCheckResult:
    """Analyse an image for manipulation and verify its alleged context."""
    if not GENAI_AVAILABLE:
        raise GeminiUnavailable(
            "The 'google-genai' package is not importable, so images cannot be sent."
        )
    caption = (claim or "").strip()
    if caption:
        instruction = (
            "ALLEGED CONTEXT SUPPLIED BY THE USER (an unverified claim, not established fact):\n"
            f"{caption}\n\n"
            "Inspect the image for manipulation, verify whether this alleged context is accurate, "
            "and return the required JSON object."
        )
    else:
        instruction = (
            "No caption or context was supplied. Identify what the image actually shows and where "
            "it comes from, assess its authenticity, and return the required JSON object."
        )
    contents: List[Any] = [
        genai_types.Part.from_bytes(data=raw, mime_type=mime),
        instruction,
    ]
    hash_key = _image_hash(raw, claim)
    hit = _cache_get(hash_key)
    if hit is not None:
        log.info("Cache HIT for image (%s)", hash_key[:12])
        return FactCheckResult(**_coerce_payload(hit))

    response = _generate(IMAGE_SYSTEM_INSTRUCTION, contents, _build_schema())
    result = _parse_result(response)
    _cache_put(hash_key, "image", result.model_dump(), query_text=caption)
    result.cached = False
    return result


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log a precise readiness report as soon as the server boots."""
    _init_cache()
    info = MANAGER.status()
    log.info(
        "Starting fact-check API | model=%s | sdk_installed=%s | api_key=%s | "
        "search_grounding=on",
        info["model"],
        info["sdk_installed"],
        "yes" if info["api_key_configured"] else "NO",
    )
    if not info["sdk_installed"]:
        log.error("google-genai is not importable: %s", GENAI_IMPORT_ERROR)
    elif not info["api_key_configured"]:
        log.warning(
            "GEMINI_API_KEY is missing or still the placeholder; /verify/* will "
            "return HTTP 503 until %s is filled in.",
            os.path.join(BASE_DIR, ".env"),
        )
    yield
    log.info("Fact-check API stopped.")


app = FastAPI(
    title="Fake News & Manipulated Image Detector",
    description=(
        "Grounded fact-checking for text claims and images. Verdicts are backed by "
        "live web search, and every response carries the real source URLs used to "
        "reach it."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_WILDCARD_CORS = "*" in CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not _WILDCARD_CORS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(GeminiUnavailable)
async def _handle_unavailable(_: Request, exc: GeminiUnavailable) -> JSONResponse:
    log.error("Gemini unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"error": "service_unavailable", "detail": str(exc)},
    )


@app.exception_handler(GeminiFailure)
async def _handle_failure(_: Request, exc: GeminiFailure) -> JSONResponse:
    log.error("Gemini failure: %s", exc)
    return JSONResponse(
        status_code=502, content={"error": "service_failure", "detail": str(exc)}
    )


@app.exception_handler(ModelOutputError)
async def _handle_bad_output(_: Request, exc: ModelOutputError) -> JSONResponse:
    log.error("Unusable model output: %s", exc)
    return JSONResponse(
        status_code=502, content={"error": "invalid_model_output", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": f"Unexpected server error: {type(exc).__name__}: {exc}",
        },
    )


# --------------------------------------------------------------------------- #
# System routes
# --------------------------------------------------------------------------- #
@app.get("/", tags=["system"], summary="Web UI (single-page app)")
def root() -> Any:
    """Serve the merged front-end. One process hosts both UI and API."""
    if os.path.isfile(INDEX_FILE):
        return FileResponse(INDEX_FILE, media_type="text/html")
    return {
        "service": "Fake News & Manipulated Image Detector",
        "model": MODEL_NAME,
        "docs": "/docs",
        "endpoints": ["/verify/text", "/verify/image", "/health", "/config"],
        "note": "static/index.html is missing; serving the API banner instead.",
    }


@app.get("/api-info", tags=["system"], summary="Machine-readable service banner")
def api_info() -> Dict[str, Any]:
    return {
        "service": "Fake News & Manipulated Image Detector",
        "model": MODEL_NAME,
        "docs": "/docs",
        "ui": "/",
        "endpoints": ["/verify/text", "/verify/image", "/health", "/config"],
    }


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="Readiness")
def health() -> HealthResponse:
    info = MANAGER.status()
    ready = bool(info["sdk_installed"] and info["api_key_configured"])
    if ready:
        detail = "Ready: the analysis engine is configured and live web search is on."
    elif not info["sdk_installed"]:
        detail = f"Degraded: the analysis engine is not available ({GENAI_IMPORT_ERROR})."
    else:
        detail = "Degraded: the analysis engine is not configured. Contact the administrator."
    return HealthResponse(
        status="ok" if ready else "degraded",
        model=info["model"],
        sdk_installed=bool(info["sdk_installed"]),
        api_key_configured=bool(info["api_key_configured"]),
        search_grounding=True,
        detail=detail,
    )


@app.get("/config", tags=["system"], summary="Non-secret configuration")
def configuration() -> Dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "search_grounding": True,
        "url_context": any(
            getattr(tool, "url_context", None) is not None
            for tool in _build_tools()
            if not isinstance(tool, dict)
        ),
        "max_text_chars": MAX_TEXT_CHARS,
        "max_image_bytes": MAX_IMAGE_BYTES,
        "allowed_image_types": sorted(ALLOWED_IMAGE_TYPES),
        "verdicts": [verdict.value for verdict in Verdict],
        "cors_origins": CORS_ORIGINS,
    }


@app.get("/trending", tags=["system"], summary="Most frequently checked claims")
def get_trending() -> Dict[str, Any]:
    """Return the top 5 most frequently queried claims (misinformation radar)."""
    results: List[Dict[str, Any]] = []
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT query_text, response_json, hit_count
                FROM verifications
                WHERE query_text IS NOT NULL AND query_text != ''
                ORDER BY hit_count DESC
                LIMIT 5
                """
            )
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        log.warning("Trending lookup failed: %s", exc)
        return {"trending": []}
    for text, resp, count in rows:
        try:
            data = json.loads(resp)
        except (TypeError, ValueError):
            data = {}
        results.append(
            {
                "claim": text,
                "verdict": data.get("verdict", "UNKNOWN"),
                "truth_percentage": data.get("truth_percentage", 0),
                "hit_count": count,
            }
        )
    return {"trending": results}


@app.get("/history", tags=["system"], summary="Recent verification history")
def get_history(limit: int = 15) -> Dict[str, Any]:
    """Return the most recent checked claims, newest first."""
    history: List[Dict[str, Any]] = []
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT hash_key, query_text, response_json, created_at
                FROM verifications
                WHERE query_text IS NOT NULL AND query_text != ''
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        log.warning("History lookup failed: %s", exc)
        return {"history": []}
    for hash_key, text, resp, created_at in rows:
        try:
            data = json.loads(resp)
        except (TypeError, ValueError):
            data = {}
        history.append(
            {
                "hash_key": hash_key,
                "claim": text,
                "verdict": data.get("verdict", "UNKNOWN"),
                "truth_percentage": data.get("truth_percentage", 0),
                "created_at": created_at,
            }
        )
    return {"history": history}


@app.delete("/history/{hash_key}", tags=["system"], summary="Delete one history item")
def delete_history_item(hash_key: str) -> Dict[str, str]:
    """Remove a single cached verification from the history."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute("DELETE FROM verifications WHERE hash_key = ?", (hash_key,))
            conn.commit()
    except sqlite3.Error as exc:
        log.warning("History delete failed for %s: %s", hash_key[:12], exc)
        raise HTTPException(status_code=500, detail="Could not delete the history item.")
    return {"status": "deleted", "hash_key": hash_key}


@app.delete("/history", tags=["system"], summary="Clear all history")
def clear_all_history() -> Dict[str, str]:
    """Remove every cached verification from the history."""
    try:
        with sqlite3.connect(CACHE_DB_PATH) as conn:
            conn.execute("DELETE FROM verifications")
            conn.commit()
    except sqlite3.Error as exc:
        log.warning("History clear failed: %s", exc)
        raise HTTPException(status_code=500, detail="Could not clear the history.")
    return {"status": "cleared"}


# --------------------------------------------------------------------------- #
# Verification routes
# --------------------------------------------------------------------------- #
@app.post(
    "/verify/text",
    response_model=FactCheckResult,
    tags=["verification"],
    summary="Fact-check a text claim or news article",
)
def verify_text_route(payload: TextVerificationRequest) -> FactCheckResult:
    """Ground a claim against live web search and return a verdict + sources."""
    log.info("Verifying text claim (%d characters)", len(payload.text))
    return verify_text(payload.text)


@app.post(
    "/verify/image",
    response_model=FactCheckResult,
    tags=["verification"],
    summary="Forensically analyse an image and its alleged context",
)
def verify_image_route(
    file: UploadFile = File(..., description="JPG, PNG or WEBP image to analyse."),
    claim: Optional[str] = Form(
        None, description="Optional caption or alleged context for the image."
    ),
) -> FactCheckResult:
    """Inspect an upload for manipulation and verify the context it is presented in."""
    raw = file.file.read()
    mime = _validate_image(raw, file.content_type)
    log.info(
        "Verifying image '%s' (%s, %d bytes, caption=%s)",
        file.filename,
        mime,
        len(raw),
        "yes" if (claim or "").strip() else "no",
    )
    return verify_image(raw, mime, claim)


# Backwards-compatible aliases so both API generations keep working.
app.add_api_route(
    "/analyze-text",
    verify_text_route,
    methods=["POST"],
    response_model=FactCheckResult,
    tags=["verification"],
    summary="Alias of /verify/text",
)
app.add_api_route(
    "/analyze-image",
    verify_image_route,
    methods=["POST"],
    response_model=FactCheckResult,
    tags=["verification"],
    summary="Alias of /verify/image",
)


# --------------------------------------------------------------------------- #
# Merged front-end (single-page app served by this same FastAPI process)
# --------------------------------------------------------------------------- #
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:  # pragma: no cover - only when the assets were deleted
    log.warning("Static directory %s not found; the web UI will not be served.", STATIC_DIR)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(
        "backend:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
