"""
frontend.py
===========
Optional Streamlit UI for the Fake News & Manipulated Image Detector.

The primary interface is the single-page app served by the FastAPI backend at
``/`` (see ``static/``). This Streamlit client mirrors the same design and talks
to the very same API, so both entry points behave identically.

Design
------
One card with a drop-image zone and a paste-text box, three quick samples, one
yellow call-to-action button, and a results panel that shows the verdict, a
truth score and clickable source URLs.

Run
---
    streamlit run frontend.py --server.port 8501
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("FRONTEND_TIMEOUT", "300"))

TEXT_ENDPOINT = "/verify/text"
IMAGE_ENDPOINT = "/verify/image"
HEALTH_ENDPOINT = "/health"
TRENDING_ENDPOINT = "/trending"
HISTORY_ENDPOINT = "/history"

YELLOW = "#FFD447"

# Verdict -> label, accent colour, badge background, emoji, explanation.
VERDICT_STYLES: Dict[str, Dict[str, str]] = {
    "TRUE": {
        "label": "TRUE / GENUINE",
        "colour": "#15803d",
        "bg": "#e9f7ee",
        "emoji": "\u2705",
        "blurb": "The claim is corroborated by reliable sources.",
    },
    "FALSE": {
        "label": "FALSE / FAKE",
        "colour": "#b42318",
        "bg": "#fdecea",
        "emoji": "\u274c",
        "blurb": "The claim is contradicted by reliable sources or fabricated.",
    },
    "PARTIALLY_TRUE": {
        "label": "PARTIALLY TRUE / MISLEADING",
        "colour": "#b54708",
        "bg": "#fff6e8",
        "emoji": "\u26a0\ufe0f",
        "blurb": "Parts of the claim are accurate; key parts are not.",
    },
    "UNVERIFIED": {
        "label": "UNVERIFIED / NO EVIDENCE",
        "colour": "#475467",
        "bg": "#f2f4f7",
        "emoji": "\u26aa",
        "blurb": "Not enough reliable information online to confirm or deny this.",
    },
}
FALLBACK_STYLE = {
    "label": "UNKNOWN",
    "colour": "#475467",
    "bg": "#f2f4f7",
    "emoji": "\u2753",
    "blurb": "",
}

# Quick samples shown as pills under the heading.
SAMPLES: Dict[str, Dict[str, str]] = {
    "Real Science News": {
        "dot": "#16a34a",
        "text": (
            "A study published in the journal Nature Climate Change found that global "
            "average sea surface temperatures in 2024 were the highest recorded since "
            "instrumental measurements began in 1850. The research was led by NOAA and "
            "independently confirmed by NASA and the UK Met Office Hadley Centre."
        ),
    },
    "Partially Fake News": {
        "dot": "#f59e0b",
        "text": (
            "The Indian government has announced that from next month every citizen will "
            "receive free unlimited 5G internet on their mobile phone. Officials said the "
            "scheme is fully funded and will be implemented nationwide within a week."
        ),
    },
    "Fake Viral News": {
        "dot": "#dc2626",
        "text": (
            "BREAKING: NASA has confirmed that the Earth will experience 15 days of "
            "complete darkness in December because of a rare alignment of Jupiter and "
            "Saturn. Forward this message to 10 people to stay safe."
        ),
    },
}

st.set_page_config(
    page_title="Fact Check - AI Fact Verification",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Theme (mirrors static/styles.css)
# --------------------------------------------------------------------------- #
THEME_CSS = f"""
<style>
  .stApp {{ background: #ffffff; }}
  .block-container {{ max-width: 1010px; padding-top: 1.5rem; }}
  header[data-testid="stHeader"] {{ background: #fffdf5; }}
  html, body, [class*="css"] {{ font-size: 17px; }}
  h1 {{ font-size: 2.4rem !important; letter-spacing: -.02em; }}
  h2 {{ font-size: 1.65rem !important; }}
  h3 {{ font-size: 1.35rem !important; }}
  p, li, label, .stMarkdown {{ font-size: 16.5px; line-height: 1.6; }}
  div.stButton > button {{
    border-radius: 14px; font-weight: 600; padding: .7rem 1rem;
    font-size: 16px; border: 1px solid #e5e7eb;
    transition: transform .12s ease, box-shadow .12s ease,
                background .12s ease, border-color .12s ease;
  }}
  div.stButton > button:hover {{
    transform: translateY(-2px); border-color: {YELLOW};
    box-shadow: 0 4px 12px rgba(17,24,39,.12); background: #fffdf5;
  }}
  div.stButton > button[kind="primary"] {{
    background: {YELLOW}; color: #111827; border: none; font-size: 18px;
    font-weight: 700; box-shadow: 0 2px 0 #e8b800;
  }}
  div.stButton > button[kind="primary"]:hover {{
    background: #ffc93c; color: #111827; transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(232,184,0,.35);
  }}
  div[data-testid="stFileUploader"] section {{
    border: 1.6px dashed {YELLOW}; background: #fffdf5; border-radius: 14px;
    transition: border-color .12s ease, background .12s ease,
                box-shadow .12s ease;
  }}
  div[data-testid="stFileUploader"] section:hover {{
    border-color: #ffc93c; background: #fffaf0;
    box-shadow: 0 4px 14px rgba(255,212,71,.28);
  }}
  textarea {{ border-radius: 14px !important; font-size: 16.5px !important; }}
  textarea:focus {{ border-color: {YELLOW} !important;
                   box-shadow: 0 0 0 3px rgba(255,212,71,.28) !important; }}
  div[data-testid="stMetricValue"] {{ font-size: 34px; font-weight: 800; }}
  div[data-testid="stMetricLabel"] {{ font-size: 15px; }}
  div[data-testid="stAlert"] {{ font-size: 16.5px; border-radius: 12px; }}
  a {{ transition: color .12s ease; }}
  a:hover {{ color: #b8860b; text-decoration: underline; }}
  div[data-testid="stExpander"] summary:hover {{ color: #b8860b; }}
</style>
"""

# --------------------------------------------------------------------------- #
# Backend helpers
# --------------------------------------------------------------------------- #
def backend_health(base_url: str) -> Optional[Dict[str, Any]]:
    """Return the backend /health payload, or None if it is unreachable."""
    try:
        response = requests.get(f"{base_url}{HEALTH_ENDPOINT}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None


def fetch_trending(base_url: str) -> List[Dict[str, Any]]:
    """Return the top checked claims from the backend radar, or an empty list."""
    try:
        response = requests.get(f"{base_url}{TRENDING_ENDPOINT}", timeout=10)
        if response.status_code == 200:
            return response.json().get("trending") or []
    except (requests.RequestException, ValueError):
        return []
    return []


def fetch_history(base_url: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Return recent verification history from the backend, or an empty list."""
    try:
        response = requests.get(
            f"{base_url}{HISTORY_ENDPOINT}", params={"limit": limit}, timeout=10
        )
        if response.status_code == 200:
            return response.json().get("history") or []
    except (requests.RequestException, ValueError):
        return []
    return []


def delete_history_item(base_url: str, hash_key: str) -> bool:
    """Delete one history entry; return True on success."""
    try:
        response = requests.delete(
            f"{base_url}{HISTORY_ENDPOINT}/{hash_key}", timeout=10
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def clear_history(base_url: str) -> bool:
    """Delete every history entry; return True on success."""
    try:
        response = requests.delete(f"{base_url}{HISTORY_ENDPOINT}", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_text_api(base_url: str, text: str) -> Dict[str, Any]:
    """POST a claim to /verify/text and return the parsed verdict payload."""
    response = requests.post(
        f"{base_url}{TEXT_ENDPOINT}", json={"text": text}, timeout=REQUEST_TIMEOUT
    )
    return _handle_response(response)


def call_image_api(
    base_url: str, filename: str, data: bytes, mime: str, claim: str
) -> Dict[str, Any]:
    """POST an image (and optional caption) to /verify/image."""
    files = {"file": (filename, data, mime)}
    form = {"claim": claim} if claim.strip() else {}
    response = requests.post(
        f"{base_url}{IMAGE_ENDPOINT}",
        files=files,
        data=form,
        timeout=REQUEST_TIMEOUT,
    )
    return _handle_response(response)


def _handle_response(response: requests.Response) -> Dict[str, Any]:
    """Turn a backend response into a payload dict, or raise a readable error."""
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"The backend returned non-JSON content ({exc}).") from exc
    detail: Any = ""
    try:
        body = response.json()
        detail = body.get("detail") or body.get("error") or str(body)
    except ValueError:
        detail = response.text[:500]
    if isinstance(detail, list):  # FastAPI validation error list
        detail = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg', '')}"
            for item in detail
            if isinstance(item, dict)
        )
    raise RuntimeError(f"HTTP {response.status_code}: {detail}")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_verdict_banner(payload: Dict[str, Any]) -> None:
    """Deprecated: the verdict is now rendered by :func:`render_result`."""
    return


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Render numbered, clickable source links."""
    st.subheader("\U0001f517 Verified Sources")
    links = [s for s in sources if str(s.get("url") or "").strip()]
    if not links:
        st.info(
            "Verified against historical and factual knowledge base "
            "(live web citations unavailable)."
        )
        return
    for index, source in enumerate(links, start=1):
        url = str(source.get("url")).strip()
        title = str(source.get("title") or url).strip()
        st.markdown(f"{index}. [{title}]({url})")


def render_result(payload: Dict[str, Any]) -> None:
    """Render the full verdict payload returned by the backend."""
    verdict = str(payload.get("verdict", "")).upper()
    style = VERDICT_STYLES.get(verdict, FALLBACK_STYLE)

    # -- prominent verdict banner ------------------------------------------ #
    banner = f"{style['emoji']} {style['label']}"
    if verdict == "TRUE":
        st.success(banner)
    elif verdict == "PARTIALLY_TRUE":
        st.warning(banner)
    elif verdict == "UNVERIFIED":
        st.info(
            "\u26aa **UNVERIFIED:** There is not much information online to back "
            "this information up."
        )
    else:
        st.error(banner)
    if payload.get("cached"):
        st.caption("\u26a1 Instant Result (Cached)")
    if style["blurb"]:
        st.caption(style["blurb"])

    # -- "What is True vs What is False" card ------------------------------ #
    st.subheader("\u2696\ufe0f What is True vs What is False")
    truth_points, false_points = _split_points(payload)
    with st.container(border=True):
        true_col, false_col = st.columns(2)
        with true_col:
            st.markdown("**\u2705 What is True**")
            if truth_points:
                for point in truth_points:
                    st.markdown(f"- {point}")
            else:
                st.markdown("_Nothing in this claim is supported by evidence._")
        with false_col:
            st.markdown("**\u274c What is False / Unverified**")
            if false_points:
                for point in false_points:
                    st.markdown(f"- {point}")
            else:
                st.markdown("_No inaccuracies were identified._")

    # -- concise analysis (propagation summary kept if present) ------------ #
    analysis = str(payload.get("analysis") or "").strip()
    if analysis:
        with st.expander("\U0001f9e0 Full analysis", expanded=False):
            st.markdown(analysis)

    render_sources(payload.get("sources") or [])


def _percentage(payload: Dict[str, Any]) -> float:
    """Return the verdict's truth percentage clamped to 0-100."""
    try:
        value = float(payload.get("truth_percentage") or 0)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(100.0, value))


def _split_points(payload: Dict[str, Any]) -> tuple:
    """Derive concise true/false bullet points from the verdict payload.

    The backend returns prose in ``analysis`` and ``details``. For a clean,
    scannable card we split those fields into short sentences and route them to
    the "true" or "false" column based on the overall verdict. Anything the
    model flagged as a correction or hoax goes to the false column.
    """
    verdict = str(payload.get("verdict", "")).upper()
    details = str(payload.get("details") or "").strip()
    analysis = str(payload.get("analysis") or "").strip()

    def bullets(text: str, limit: int = 6) -> List[str]:
        points: List[str] = []
        for sentence in _sentences(text):
            points.append(sentence)
            if len(points) >= limit:
                break
        return points

    correction = str(payload.get("correction") or "").strip()
    if verdict == "TRUE":
        truth_points = bullets(details) or bullets(analysis)
        false_points: List[str] = []
    elif verdict == "UNVERIFIED":
        truth_points = []
        false_points = []
    elif verdict == "FALSE":
        truth_points = bullets(correction)
        false_points = bullets(details) or bullets(analysis)
    else:  # PARTIALLY_TRUE
        truth_points = bullets(correction)
        false_points = bullets(details) or bullets(analysis)
    return truth_points, false_points


def _sentences(text: str, max_len: int = 240) -> List[str]:
    """Split prose into short, trimmed sentence bullets."""
    import re

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    raw = re.split(r"(?<=[.!?])\s+", cleaned)
    out: List[str] = []
    for piece in raw:
        piece = piece.strip(" -•\t")
        if len(piece) < 3:
            continue
        if len(piece) > max_len:
            piece = piece[: max_len - 1].rstrip() + "\u2026"
        out.append(piece)
    return out


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header() -> str:
    """Draw the branded top bar and return the backend base URL to use."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    st.title("\U0001f50d FACT CHECK")
    st.caption("AI Fact Verification \u00b7 live web analysis")

    base_url = DEFAULT_BACKEND
    health = backend_health(base_url)
    if health is None:
        st.error(
            f"Backend unreachable at {base_url}. Start it with "
            "`uvicorn backend:app --port 8000` (or `python app.py`)."
        )
    elif health.get("status") != "ok":
        st.warning(health.get("detail") or "Backend degraded.")
    elif health.get("api_key_configured", True):
        st.success("\u2705 System Ready \u00b7 Live Web Search Enabled")
    if health and not health.get("api_key_configured", True):
        st.warning("System not ready. Contact the administrator to enable analysis.")
    return base_url



# --------------------------------------------------------------------------- #
# Main form (single card: drop image + paste text + one CTA)
# --------------------------------------------------------------------------- #
def main_form(base_url: str) -> None:
    """One card with an image drop zone, a text box and the yellow CTA."""
    st.subheader("Paste or upload the news article which needs to be checked")
    st.caption(
        "Check if news is True, Partially Fake, or Fake. Get a percentage "
        "legitimacy score, full source URLs, and the verified truth."
    )

    # ---- quick samples ---------------------------------------------------- #
    st.caption("Quick Samples:")
    sample_cols = st.columns([1, 1, 1])
    for column, (name, payload) in zip(sample_cols, SAMPLES.items()):
        with column:
            if st.button(name, key=f"sample_{name}", use_container_width=True):
                st.session_state["text_claim"] = payload["text"]
                st.session_state.pop("text_result", None)
                st.session_state.pop("image_result", None)
                st.rerun()

    st.write("")

    # ---- the input card --------------------------------------------------- #
    with st.container(border=True):
        left, right = st.columns(2)

        with left:
            st.markdown("**\U0001f5bc\ufe0f Drop Image Screenshot** _(optional)_")
            upload = st.file_uploader(
                "Drag & drop screenshot here, or browse. Supports PNG, JPG, WebP.",
                type=["jpg", "jpeg", "png", "webp"],
                key="image_upload",
                label_visibility="visible",
            )
            if upload is not None:
                st.image(upload, caption=upload.name, use_container_width=True)

        with right:
            claim = st.text_area(
                "\U0001f4c4 PASTE NEWS ARTICLE TEXT",
                height=258,
                placeholder=(
                    "Paste news headline, article paragraphs, social media post, "
                    "or claim here..."
                ),
                key="text_claim",
            )
            words = len(claim.split())
            st.caption(f"{words} " + ("word" if words == 1 else "words"))

        check = st.button(
            "\U0001f50e Check if News is Legit or Fake",
            type="primary",
            use_container_width=True,
        )

    if check:
        _run_check(base_url, claim, upload)

    if st.session_state.get("image_result"):
        st.divider()
        render_result(st.session_state["image_result"])
    elif st.session_state.get("text_result"):
        st.divider()
        render_result(st.session_state["text_result"])


def _run_check(base_url: str, claim: str, upload: Any) -> None:
    """Send whichever input the user supplied and store the verdict."""
    text = (claim or "").strip()
    if not text and upload is None:
        st.warning("Paste an article or attach a screenshot before checking.")
        return

    with st.spinner("Searching the live web and fact-checking..."):
        try:
            if upload is not None:
                payload = call_image_api(
                    base_url,
                    upload.name,
                    upload.getvalue(),
                    upload.type or "application/octet-stream",
                    text,
                )
                st.session_state["image_result"] = payload
                st.session_state.pop("text_result", None)
            else:
                payload = call_text_api(base_url, text)
                st.session_state["text_result"] = payload
                st.session_state.pop("image_result", None)
        except RuntimeError as exc:
            st.session_state.pop("text_result", None)
            st.session_state.pop("image_result", None)
            st.error(str(exc))
        except requests.RequestException as exc:
            st.session_state.pop("text_result", None)
            st.session_state.pop("image_result", None)
            st.error(f"Could not reach the backend at {base_url}: {exc}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_trending_radar(base_url: str) -> None:
    """Show the most frequently checked claims in the sidebar (radar)."""
    trending = fetch_trending(base_url)
    with st.sidebar:
        st.markdown("### 🔥 Misinformation Radar")
        st.caption("Most frequently checked claims")
        if not trending:
            st.caption("No trending claims yet. Verify something to start the radar.")
            return
        for item in trending:
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            verdict = str(item.get("verdict") or "UNKNOWN").upper()
            style = VERDICT_STYLES.get(verdict, FALLBACK_STYLE)
            count = int(item.get("hit_count") or 0)
            preview = claim if len(claim) <= 110 else claim[:107].rstrip() + "..."
            st.markdown(
                f'<div style="border:1px solid #e5e7eb;border-left:5px solid '
                f'{style["colour"]};border-radius:10px;padding:10px 12px;'
                f'margin-bottom:10px;background:{style["bg"]};">'
                f'<div style="font-size:14px;font-weight:700;color:{style["colour"]};">'
                f'{style["emoji"]} {style["label"]}</div>'
                f'<div style="font-size:13px;color:#374151;margin:4px 0 6px;">{preview}</div>'
                f'<div style="font-size:12px;color:#6b7280;">🔍 Checked {count} time(s)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_history_section(base_url: str) -> None:
    """Show recent checks in the sidebar with delete + clear-all controls."""
    history = fetch_history(base_url)
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🗂️ Search History")
        if not history:
            st.caption("No history yet. Your checked claims will appear here.")
            return

        if st.button("🧹 Clear All History", key="clear_history_btn"):
            if clear_history(base_url):
                st.success("History cleared.")
                st.rerun()
            else:
                st.error("Could not clear history.")

        for item in history:
            hash_key = str(item.get("hash_key") or "").strip()
            claim = str(item.get("claim") or "").strip()
            if not hash_key or not claim:
                continue
            verdict = str(item.get("verdict") or "UNKNOWN").upper()
            style = VERDICT_STYLES.get(verdict, FALLBACK_STYLE)
            preview = claim if len(claim) <= 90 else claim[:87].rstrip() + "..."
            created = str(item.get("created_at") or "")

            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(
                    f'<div style="border:1px solid #e5e7eb;border-left:5px solid '
                    f'{style["colour"]};border-radius:10px;padding:8px 10px;'
                    f'margin-bottom:6px;background:{style["bg"]};">'
                    f'<div style="font-size:13px;font-weight:700;color:{style["colour"]};">'
                    f'{style["emoji"]} {style["label"]}</div>'
                    f'<div style="font-size:12.5px;color:#374151;margin:4px 0 4px;">'
                    f'{preview}</div>'
                    f'<div style="font-size:11px;color:#9ca3af;">{created}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("🗑️", key=f"del_{hash_key}", help="Delete this entry"):
                    if delete_history_item(base_url, hash_key):
                        st.rerun()
                    else:
                        st.error("Delete failed.")


def main() -> None:
    """Compose the page: branded header plus the single verification card."""
    base_url = render_header()
    render_trending_radar(base_url)
    render_history_section(base_url)
    main_form(base_url)

    st.markdown(
        '<div style="text-align:center;color:#6b7280;font-size:13px;'
        'border-top:1px solid #e5e7eb;margin-top:36px;padding-top:16px;">'
        "Verdicts are AI-generated using live web analysis. "
        "Always verify cited sources before acting on a verdict.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
else:
    # Streamlit executes the module top-to-bottom, so render on import too.
    main()
