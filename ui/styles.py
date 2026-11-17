"""
Custom CSS Styles and Themes for Academic Presentation & Viva Agent
"""

CUSTOM_CSS = """
<style>
/* Main Layout Tweaks */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Header Banner */
.academic-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
    padding: 1.8rem 2rem;
    border-radius: 12px;
    color: #f8fafc;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
}

.academic-header h1 {
    color: #ffffff !important;
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.academic-header p {
    color: #cbd5e1 !important;
    font-size: 1.05rem;
    margin-bottom: 0;
}

/* Criteria Badge */
.criteria-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

.badge-passed {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
}

/* Card Container */
.academic-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Slide Preview Card */
.slide-card {
    background: #ffffff;
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1);
}

.slide-header {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
}

.slide-tag {
    color: #2563eb;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.slide-title {
    color: #0f172a;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 0.2rem;
}

.slide-bullet {
    margin-bottom: 0.5rem;
    font-size: 1.0rem;
    color: #334155;
    line-height: 1.5;
}

.slide-takeaway {
    background-color: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin-top: 1rem;
    font-weight: 600;
    color: #92400e;
    font-size: 0.95rem;
}

.speaker-notes-box {
    background-color: #f8fafc;
    border: 1px dashed #94a3b8;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-top: 1rem;
    color: #475569;
    font-size: 0.9rem;
    font-style: italic;
}

/* Viva Flashcard */
.viva-card {
    background: #ffffff;
    border-left: 5px solid #8b5cf6;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.viva-tier-badge {
    background: #ede9fe;
    color: #6d28d9;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
}

.viva-trap {
    background-color: #fee2e2;
    border-left: 3px solid #ef4444;
    padding: 0.6rem 0.85rem;
    border-radius: 4px;
    color: #991b1b;
    font-size: 0.88rem;
    margin-top: 0.6rem;
}

/* Intent Router Trace Pill */
.intent-pill {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    padding: 0.35rem 0.8rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
</style>
"""
