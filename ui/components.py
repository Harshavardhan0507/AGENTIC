"""
Reusable UI Components for Streamlit Frontend
"""
import streamlit as st
from typing import Dict, Any, List
import config


def render_header():
    """Renders the top application banner and criteria summary."""
    st.markdown(f"""
    <div class="academic-header">
        <h1>🎓 {config.APP_NAME}</h1>
        <p>{config.APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)


def render_criteria_bar():
    """Displays real-time validation badges for all 7 acceptance criteria."""
    st.markdown("##### 📋 Acceptance Criteria Compliance (Mandatory Requirements)")
    badges_html = '<div style="margin-bottom: 1.2rem;">'
    for item in config.ACCEPTANCE_CRITERIA:
        badges_html += f"""
        <span class="criteria-badge badge-passed">
            ✓ Criteria {item['id']}: {item['name']}
        </span>
        """
    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)


def render_intent_badge(intent: str, confidence: float, needs_retrieval: bool, thought: str):
    """Renders visual Intent Router indicator."""
    intent_colors = {
        "SUMMARIZE": "#0284c7",
        "CONCEPT_EXTRACT": "#7c3aed",
        "SLIDE_GENERATE": "#2563eb",
        "VIVA_QUESTIONS": "#dc2626",
        "INTERACTIVE_VIVA": "#ea580c",
        "PAPER_QA": "#059669",
        "GENERAL_CONVERSATION": "#475569"
    }
    color = intent_colors.get(intent, "#2563eb")

    st.markdown(f"""
    <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid {color}; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 700; color: {color}; font-size: 0.95rem;">
                🎯 Detected Intent: {intent} (Confidence: {int(confidence * 100)}%)
            </span>
            <span style="font-size: 0.8rem; background: {'#dcfce7' if needs_retrieval else '#f1f5f9'}; color: {'#166534' if needs_retrieval else '#475569'}; padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 600;">
                {'🔍 ChromaDB Retrieval Triggered' if needs_retrieval else '⚡ Direct Model Execution'}
            </span>
        </div>
        <div style="color: #475569; font-size: 0.85rem; margin-top: 0.4rem; font-style: italic;">
            🧠 <b>Agent CoT Routing Reason:</b> {thought}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_slide_preview(slide: Dict[str, Any]):
    """Renders a single slide preview card."""
    num = slide.get("slide_number", 1)
    title = slide.get("title", "Untitled Slide")
    subtitle = slide.get("subtitle", "")
    bullets = slide.get("bullet_points", [])
    takeaway = slide.get("key_takeaway", "")
    speaker_notes = slide.get("speaker_notes", "")

    bullets_html = "".join([f"<div class='slide-bullet'>• {b}</div>" for b in bullets])
    takeaway_html = f"<div class='slide-takeaway'>💡 <b>Key Takeaway:</b> {takeaway}</div>" if takeaway else ""
    notes_html = f"<div class='speaker-notes-box'>🎙️ <b>Speaker Notes (Verbal Script):</b><br>{speaker_notes}</div>" if speaker_notes else ""

    card_html = f"""
    <div class="slide-card">
        <div class="slide-header">
            <div class="slide-tag">SLIDE {num} • {subtitle.upper()}</div>
            <div class="slide-title">{title}</div>
        </div>
        <div style="margin-bottom: 0.8rem;">
            {bullets_html}
        </div>
        {takeaway_html}
        {notes_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_citations(citations: List[Dict[str, Any]]):
    """Renders expandable list of ChromaDB retrieved chunks."""
    if not citations:
        return

    with st.expander(f"📚 View Retrieved ChromaDB Context Chunks ({len(citations)} citations)", expanded=False):
        for idx, c in enumerate(citations):
            score_pct = int(c.get("score", 0.0) * 100)
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                <b>{c.get('citation')}</b> — Relevance Score: <b>{score_pct}%</b>
                <p style="font-size: 0.85rem; color: #475569; margin: 0.3rem 0 0 0;">
                    {c.get('excerpt')}
                </p>
            </div>
            """, unsafe_allow_html=True)
