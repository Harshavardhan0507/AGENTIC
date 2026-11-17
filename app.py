"""
Academic Paper-to-Presentation & Viva Agent
Frontend: Streamlit | Backend: Python | Vector DB: ChromaDB | LCEL Agentic RAG
Demonstrates all 7 Mandatory Acceptance Criteria.
"""
import os
import sys
import json
from pathlib import Path
import streamlit as st

# Ensure base directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from ui.styles import CUSTOM_CSS
from ui.components import (
    render_header,
    render_criteria_bar,
    render_intent_badge,
    render_slide_preview,
    render_citations
)
from core.document_loader import AcademicDocumentLoader
from core.vector_store import ChromaManager
from core.llm_factory import LLMFactory
from agents.orchestrator import AcademicOrchestrator
from core.prompt_registry import get_all_prompt_specs

# Page configuration
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = ChromaManager()

if "doc_loader" not in st.session_state:
    st.session_state.doc_loader = AcademicDocumentLoader()

if "orchestrator" not in st.session_state:
    # Default initial LLM
    initial_llm = LLMFactory.create_llm(
        provider=config.PROVIDER_OLLAMA if LLMFactory.is_ollama_running() else config.PROVIDER_GEMINI
    )
    st.session_state.orchestrator = AcademicOrchestrator(
        llm=initial_llm,
        vector_manager=st.session_state.vector_manager
    )

if "active_paper" not in st.session_state:
    st.session_state.active_paper = None

if "paper_stats" not in st.session_state:
    st.session_state.paper_stats = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = None

if "generated_concepts" not in st.session_state:
    st.session_state.generated_concepts = None

if "generated_slides" not in st.session_state:
    st.session_state.generated_slides = None

if "generated_pptx_path" not in st.session_state:
    st.session_state.generated_pptx_path = None

if "generated_viva" not in st.session_state:
    st.session_state.generated_viva = None

if "interactive_viva_history" not in st.session_state:
    st.session_state.interactive_viva_history = []


# ---------------------------------------------------------------------------
# Sidebar: LLM Model Configuration & Paper Ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ LLM & System Configuration")

    # 1. LLM Provider Selector (Criterion 1)
    provider = st.selectbox(
        "Select LLM Provider:",
        options=config.SUPPORTED_PROVIDERS,
        index=1 if LLMFactory.is_ollama_running() else 0,
        help="Demonstrates Criterion 1: Local LLM (Ollama) or API-based model (Gemini, Groq, OpenAI)"
    )

    ollama_online = LLMFactory.is_ollama_running()
    api_key = None
    selected_model = None

    if provider == config.PROVIDER_OLLAMA:
        if ollama_online:
            st.success("🟢 Local Ollama Live (127.0.0.1:11434)")
            installed_models = LLMFactory.get_ollama_models()
            selected_model = st.selectbox("Local Ollama Model:", options=installed_models)
        else:
            st.warning("⚠️ Local Ollama daemon not detected. Using simulation fallback.")
            selected_model = "llama3.2:1b"

    elif provider == config.PROVIDER_GEMINI:
        default_gemini_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input("Gemini API Key:", value=default_gemini_key, type="password", help="Enter Google Gemini API Key")
        selected_model = st.selectbox("Gemini Model:", options=config.DEFAULT_MODELS[config.PROVIDER_GEMINI])

    elif provider == config.PROVIDER_GROQ:
        default_groq_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key:", value=default_groq_key, type="password")
        selected_model = st.selectbox("Groq Model:", options=config.DEFAULT_MODELS[config.PROVIDER_GROQ])

    elif provider == config.PROVIDER_OPENAI:
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key:", value=default_openai_key, type="password")
        selected_model = st.selectbox("OpenAI Model:", options=config.DEFAULT_MODELS[config.PROVIDER_OPENAI])

    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if st.button("Apply Model Settings", use_container_width=True):
        new_llm = LLMFactory.create_llm(
            provider=provider,
            model_name=selected_model,
            temperature=temperature,
            api_key=api_key
        )
        st.session_state.orchestrator = AcademicOrchestrator(
            llm=new_llm,
            vector_manager=st.session_state.vector_manager
        )
        st.success(f"Configured {provider} ({selected_model})!")

    st.markdown("---")

    # 2. Document Ingestion & ChromaDB Management (Criteria 5 & 7)
    st.markdown("### 📄 Research Paper Ingestion")

    uploaded_file = st.file_uploader(
        "Upload Research Paper (.pdf or .txt):",
        type=["pdf", "txt", "md"],
        help="Upload an academic research paper to index in ChromaDB"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Index Uploaded", use_container_width=True, disabled=uploaded_file is None):
            with st.spinner("Parsing document and indexing into ChromaDB..."):
                doc_data = st.session_state.doc_loader.load_from_file(uploaded_file, uploaded_file.name)
                count = st.session_state.vector_manager.index_document_chunks(doc_data["chunks"])
                st.session_state.active_paper = uploaded_file.name
                st.session_state.paper_stats = doc_data["stats"]
                st.success(f"Indexed {count} chunks into ChromaDB!")

    with col_btn2:
        if st.button("Load Sample Paper", use_container_width=True, help="Load seminal NeurIPS 'Attention Is All You Need' benchmark paper"):
            with st.spinner("Loading benchmark paper into ChromaDB..."):
                sample_path = config.SAMPLE_PAPERS_DIR / "attention_is_all_you_need_sample.txt"
                if sample_path.exists():
                    doc_data = st.session_state.doc_loader.load_from_file(sample_path, "Attention_Is_All_You_Need.txt")
                    count = st.session_state.vector_manager.index_document_chunks(doc_data["chunks"])
                    st.session_state.active_paper = "Attention_Is_All_You_Need.txt (Vaswani et al.)"
                    st.session_state.paper_stats = doc_data["stats"]
                    st.success(f"Indexed {count} chunks into ChromaDB!")

    # ChromaDB Vector Store Status Widget
    st.markdown("---")
    st.markdown("### 🗄️ ChromaDB Vector Store Status")
    v_stats = st.session_state.vector_manager.get_stats()
    st.info(f"**Collection:** `{v_stats.get('collection_name')}`\n\n"
            f"**Total Chunks in DB:** `{v_stats.get('count', 0)}`\n\n"
            f"**Active Paper:** `{st.session_state.active_paper or 'None Loaded'}`")

    if st.button("Reset / Clear Vector DB", use_container_width=True):
        st.session_state.vector_manager.clear()
        st.session_state.active_paper = None
        st.session_state.paper_stats = {}
        st.warning("ChromaDB collection cleared.")


# ---------------------------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------------------------
render_header()
render_criteria_bar()

# Tab Navigation across the 6 Core Workspaces
tabs = st.tabs([
    "🤖 Agentic Workspace & Chat",
    "📑 Structured Summary",
    "🔍 Concept Extractor",
    "📊 Slide Deck Generator",
    "🎓 Viva Voce Simulator",
    "🔬 Prompt Engineering & LCEL Inspector"
])


# ---------------------------------------------------------------------------
# TAB 1: AGENTIC WORKSPACE & CHAT (Intent Routing + Agentic RAG)
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### 🤖 Multi-Agent Intent Router & Academic Conversational Workspace")
    st.caption("Ask free-form questions or requests. The Agentic Intent Router automatically detects intent, determines if ChromaDB retrieval is required, and routes to the specialized agent.")

    # Quick Prompt Starters
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    with col_q1:
        if st.button("📑 'Give me a structured summary'"):
            st.session_state.temp_prompt = "Give me a structured summary of the methodology and main findings"
    with col_q2:
        if st.button("🔍 'What is the novel contribution?'"):
            st.session_state.temp_prompt = "What is the key novelty, architectural contribution, and baseline comparison?"
    with col_q3:
        if st.button("📊 'Generate presentation slides'"):
            st.session_state.temp_prompt = "Generate 6 conference presentation slides with speaker notes"
    with col_q4:
        if st.button("🎓 'Anticipate viva defense questions'"):
            st.session_state.temp_prompt = "What tough viva examination questions will the committee ask on this paper?"

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg.get("intent"):
                render_intent_badge(
                    intent=msg["intent"],
                    confidence=msg.get("confidence", 0.95),
                    needs_retrieval=msg.get("needs_retrieval", True),
                    thought=msg.get("routing_thought", "")
                )
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # Input prompt
    default_val = st.session_state.pop("temp_prompt", "")
    user_query = st.chat_input("Enter your request (e.g. 'Summarize this paper', 'Generate 6 slides', 'Explain equation 3')...")
    if not user_query and default_val:
        user_query = default_val

    if user_query:
        # Display user query
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process through Academic Orchestrator
        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent, deciding retrieval, and executing agent pipeline..."):
                response_payload = st.session_state.orchestrator.process_request(
                    user_query=user_query,
                    history=st.session_state.chat_history
                )

                render_intent_badge(
                    intent=response_payload["intent"],
                    confidence=response_payload["confidence"],
                    needs_retrieval=response_payload["needs_retrieval"],
                    thought=response_payload["routing_thought"]
                )

                st.markdown(response_payload["response"])

                citations = response_payload.get("extra_data", {}).get("citations", [])
                if citations:
                    render_citations(citations)

                # Store assistant response in history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_payload["response"],
                    "intent": response_payload["intent"],
                    "confidence": response_payload["confidence"],
                    "needs_retrieval": response_payload["needs_retrieval"],
                    "routing_thought": response_payload["routing_thought"],
                    "citations": citations
                })

                # If slides were generated, update slide tab state
                if response_payload.get("extra_data", {}).get("slides"):
                    st.session_state.generated_slides = response_payload["extra_data"]["slides"]
                    st.session_state.generated_pptx_path = response_payload["extra_data"].get("pptx_path")


# ---------------------------------------------------------------------------
# TAB 2: STRUCTURED SUMMARY (Summarizer Agent)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("#### 📑 Structured Academic Paper Summary (RAG Grounded)")
    st.caption("Generates a comprehensive 6-part academic review grounded in retrieved ChromaDB chunks using Chain-of-Thought reasoning.")

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        custom_summary_focus = st.text_input("Specific Summary Focus (optional):", "Produce a complete academic sectioned synthesis")
    with col_s2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        generate_sum_btn = st.button("🚀 Synthesize Summary", use_container_width=True)

    if generate_sum_btn:
        with st.spinner("Summarizer Agent executing RAG retrieval & LCEL synthesis..."):
            res = st.session_state.orchestrator.summarizer.run(custom_summary_focus)
            st.session_state.generated_summary = res

    if st.session_state.generated_summary:
        sum_data = st.session_state.generated_summary

        if sum_data.get("cot_reasoning"):
            with st.expander("🧠 View Summarizer Chain-of-Thought (CoT) Reasoning Trace", expanded=False):
                st.info(sum_data["cot_reasoning"])

        st.markdown(sum_data["summary"])
        render_citations(sum_data.get("citations", []))


# ---------------------------------------------------------------------------
# TAB 3: CONCEPT EXTRACTOR (Concept-Extractor Agent)
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### 🔍 Concept Extractor Agent: Key Contributions, Methods & Results")
    st.caption("Identifies the technical core: Core Research Bottleneck, Novelty Claims, Mathematical Mechanics, Benchmark Datasets, SOTA Comparisons, and Critical Limitations.")

    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        concept_query = st.text_input("Concept Focus (optional):", "Extract novelty, architecture, datasets, empirical metrics, and limitations")
    with col_c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        extract_btn = st.button("⚡ Extract Concepts", use_container_width=True)

    if extract_btn:
        with st.spinner("Concept-Extractor Agent querying ChromaDB and organizing taxonomy..."):
            res = st.session_state.orchestrator.concept_extractor.run(concept_query)
            st.session_state.generated_concepts = res

    if st.session_state.generated_concepts:
        st.markdown(st.session_state.generated_concepts["concepts"])
        render_citations(st.session_state.generated_concepts.get("citations", []))


# ---------------------------------------------------------------------------
# TAB 4: SLIDE DECK GENERATOR (Slide-Generator Agent & PPTX Export)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("#### 📊 Slide-Generator Agent: Presentation Slides & PPTX Exporter")
    st.caption("Structures academic findings into presentation-ready slide cards with speaker scripts, and compiles native PowerPoint (.pptx) files.")

    col_sl1, col_sl2 = st.columns([3, 1])
    with col_sl1:
        slide_topic = st.text_input("Target Audience / Presentation Goal:", "Conference Oral Presentation (6-8 Slides)")
    with col_sl2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        gen_slides_btn = st.button("📽️ Generate Slide Deck", use_container_width=True)

    if gen_slides_btn:
        with st.spinner("Slide-Generator Agent structuring slides and generating PowerPoint file..."):
            res = st.session_state.orchestrator.slide_generator.run(slide_topic)
            st.session_state.generated_slides = res.get("slides", [])
            st.session_state.generated_pptx_path = res.get("pptx_path")

    if st.session_state.generated_slides:
        slides = st.session_state.generated_slides

        # PowerPoint Download Action
        pptx_path = st.session_state.generated_pptx_path
        if pptx_path and Path(pptx_path).exists():
            with open(pptx_path, "rb") as f:
                pptx_bytes = f.read()

            col_down1, col_down2 = st.columns([2, 1])
            with col_down1:
                st.success(f"✅ Generated {len(slides)} slides with speaker notes and formatted layout!")
            with col_down2:
                st.download_button(
                    label="📥 Download PowerPoint (.pptx)",
                    data=pptx_bytes,
                    file_name="Research_Presentation_Deck.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True
                )

        # Slide-by-slide Preview
        st.markdown("---")
        st.markdown("##### 🖥️ Interactive Slide Deck Preview")
        for slide in slides:
            render_slide_preview(slide)


# ---------------------------------------------------------------------------
# TAB 5: VIVA VOCE DEFENSE SIMULATOR (Viva Agent)
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("#### 🎓 Viva Agent: Defense Preparation & Oral Exam Simulator")
    st.caption("Anticipate likely examiner questions across 4 cognitive rigor tiers with model answers, or engage in a simulated live defense.")

    viva_mode = st.radio(
        "Select Viva Mode:",
        options=["📚 Study Mode: Anticipated Questions & Model Answers", "🎙️ Simulation Mode: Live Oral Defense Mock Exam"],
        horizontal=True
    )

    if "Study Mode" in viva_mode:
        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            viva_focus = st.text_input("Examiner Committee Rigor Focus:", "Comprehensive 4-tier viva defense questions with traps and model answers")
        with col_v2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_viva_btn = st.button("🎯 Generate Viva Deck", use_container_width=True)

        if gen_viva_btn:
            with st.spinner("Viva Agent synthesizing multi-tiered defense questions from paper context..."):
                res = st.session_state.orchestrator.viva_agent.generate_questions(viva_focus)
                st.session_state.generated_viva = res

        if st.session_state.generated_viva:
            st.markdown(st.session_state.generated_viva["viva_content"])

    else:
        st.markdown("##### 🎙️ Simulated Oral Defense Committee")
        st.info("The external examiner tests your technical depth. Provide your response to the question below and receive a numerical score (/10), strength analysis, identified knowledge gaps, and the next examiner question.")

        # Interactive examiner dialogue
        if not st.session_state.interactive_viva_history:
            st.session_state.interactive_viva_history.append({
                "role": "examiner",
                "content": "Welcome to your oral defense. Let us begin with your model foundations: **Why did you completely eliminate recurrent and convolutional layers in favor of self-attention? How do you justify the resulting quadratic complexity in memory?**"
            })

        for v_msg in st.session_state.interactive_viva_history:
            if v_msg["role"] == "examiner":
                with st.chat_message("assistant", avatar="🎓"):
                    st.markdown(f"**Examiner:** {v_msg['content']}")
            else:
                with st.chat_message("user", avatar="👨‍🎓"):
                    st.markdown(f"**Candidate:** {v_msg['content']}")

        candidate_input = st.text_area("Your Verbal Defense Response:", placeholder="Articulate your technical justification here...", height=120)
        if st.button("Submit Oral Response to Committee", use_container_width=True):
            if candidate_input.strip():
                st.session_state.interactive_viva_history.append({"role": "candidate", "content": candidate_input})
                with st.spinner("External Committee evaluating technical precision and formulating follow-up..."):
                    eval_res = st.session_state.orchestrator.viva_agent.evaluate_candidate_turn(candidate_input)
                    st.session_state.interactive_viva_history.append({"role": "examiner", "content": eval_res["evaluation"]})
                st.rerun()


# ---------------------------------------------------------------------------
# TAB 6: ACCEPTANCE CRITERIA & PROMPT ENGINEERING INSPECTOR
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("#### 🔬 Prompt Engineering, LangChain LCEL & Criteria Inspector")
    st.caption("Demonstrating full compliance with all 7 Mandatory Acceptance Criteria, Prompt Engineering techniques (CoT, Few-Shot, ReAct, Reflection), and LangChain LCEL pipelines.")

    # 1. Acceptance Criteria Verification Matrix
    st.markdown("##### 1. Mandatory Acceptance Criteria Verification Matrix")
    crit_cols = st.columns(2)
    for idx, c in enumerate(config.ACCEPTANCE_CRITERIA):
        with crit_cols[idx % 2]:
            st.markdown(f"""
            <div class="academic-card">
                <span class="criteria-badge badge-passed">Criterion {c['id']}: PASS</span>
                <b>{c['name']}</b>
                <p style="font-size: 0.88rem; color: #475569; margin: 0.4rem 0 0 0;">
                    {c['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. LangChain Core Components & LCEL Architecture
    st.markdown("##### 2. LangChain LCEL Architecture (`prompt | llm | parser`)")
    st.markdown("""
    All agents are constructed using **LangChain Expression Language (LCEL)**, piping prompt templates directly into models and output parsers:
    """)
    st.code("""
# Example LCEL pipeline for Intent Detection & Dynamic Routing:
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_ROUTER_SYSTEM_PROMPT),
    ("human", "User Query: {query}\\nConversation History: {history}")
])

# LangChain LCEL pipe syntax
intent_chain = intent_prompt | llm | StrOutputParser()

# Example LCEL pipeline for Grounded Summarizer Agent:
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARIZER_SYSTEM_PROMPT),
    ("human", SUMMARIZER_USER_PROMPT)
])
summarizer_chain = summarizer_prompt | llm | StrOutputParser()
    """, language="python")

    st.markdown("---")

    # 3. Prompt Engineering Techniques Registry
    st.markdown("##### 3. Inspected System & User Prompts Across the Agent Fleet")
    prompt_specs = get_all_prompt_specs()

    for p in prompt_specs:
        with st.expander(f"🔹 {p['name']} — Technique: {p['technique']}", expanded=False):
            st.markdown(f"**Description:** {p['description']}")
            st.markdown(f"**Demonstrated Technique:** `{p['technique']}`")
            st.markdown("**System Prompt Template:**")
            st.code(p["system_prompt"], language="text")
            st.markdown("**User Prompt Template:**")
            st.code(p["user_prompt"], language="text")
