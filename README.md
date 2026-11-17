# Academic Paper-to-Presentation & Viva Agent

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20LCEL-green.svg)](https://www.langchain.com)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-orange.svg)](https://ollama.ai)

> **Assignment 5: Paper-to-Presentation & Viva Agent**  
> An academic assistant that takes a research paper and produces a structured summary, an auto-generated slide deck of key concepts, and a set of likely viva/examiner questions with model answers.

---

## 🎯 Problem Statement
Preparing presentations and anticipating viva questions from research papers is time-consuming for both students and faculty. This system automates the end-to-end academic preparation workflow using an **Agentic RAG architecture**, multi-agent intent routing, and grounded vector retrieval over ChromaDB.

---

## 🏛️ Multi-Agent Architecture

```
                                  ┌───────────────────────────────┐
                                  │      Streamlit Frontend       │
                                  └──────────────┬────────────────┘
                                                 │
                                                 ▼
                                  ┌───────────────────────────────┐
                                  │ Academic Intent Router (LCEL) │
                                  │  - Intent Classification      │
                                  │  - CoT Retrieval Decision     │
                                  └──────────────┬────────────────┘
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  │                                                             │
                  ▼                                                             ▼
       ┌─────────────────────┐                                       ┌─────────────────────┐
       │ Needs Retrieval?    │ ─── Query: ChromaDB Vector DB ────►   │ Grounded Chunks &   │
       │ Formulate Search    │                                       │ Citations Attached  │
       └─────────────────────┘                                       └──────────┬──────────┘
                  │                                                             │
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │
        ┌───────────────────┬────────────────────┼───────────────────┬───────────────────┐
        ▼                   ▼                    ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌────────────────┐   ┌───────────────┐   ┌───────────────┐
│  Summarizer   │   │    Concept    │   │ Slide Generator│   │  Viva Defense │   │   Paper QA    │
│     Agent     │   │Extractor Agent│   │     Agent      │   │     Agent     │   │     Agent     │
├───────────────┤   ├───────────────┤   ├────────────────┤   ├───────────────┤   ├───────────────┤
│ • 6-part RAG  │   │ • Novelty     │   │ • 6-8 Slides   │   │ • 4-tier Viva │   │ • Targeted    │
│   academic    │   │ • Architecture│   │ • Speaker Notes│   │ • Model Ans.  │   │   technical   │
│   synthesis   │   │ • Datasets    │   │ • One-Click    │   │ • Traps/Common│   │   grounded    │
│ • CoT trace   │   │ • Metrics     │   │   PowerPoint   │   │   Mistakes    │   │   answers     │
│ • Citations   │   │ • Limitations │   │   Export(.pptx)│   │ • Mock Exam   │   │ • Citations   │
└───────────────┘   └───────────────┘   └────────────────┘   └───────────────┘   └───────────────┘
```

---

## 📋 Mandatory Acceptance Criteria Verification

All 7 mandatory acceptance criteria are fully demonstrated and verified in this application:

| # | Acceptance Criterion | Implementation Details | Verified In App |
|---|----------------------|------------------------|-----------------|
| **1** | **LLM Integration** | • Local LLM: `Ollama` running `llama3.2:1b` / `llama3.2:3b` / `qwen2.5`<br>• API Providers: `Google Gemini` (`gemini-1.5-flash`), `Groq`, and `OpenAI`. Seamless switching via sidebar. | Sidebar Model Selector & Local Ollama Status |
| **2** | **Interactive GUI** | Streamlit web application with 6 dedicated tabs: Agent Workspace, Summary, Concept Extractor, Slide Deck Generator with PowerPoint export, Viva Voce simulator, and Criteria Inspector. | `app.py` Streamlit UI |
| **3** | **Prompt Engineering** | • **Chain of Thought (CoT):** `<thinking>` reasoning in Intent Routing & Summaries<br>• **Few-Shot Prompting:** Exemplars for Slide Generation & Routing<br>• **ReAct:** Reason + Act loop for ChromaDB retrieval<br>• **Reflection:** Fact-checking and grounding critique | Tab 6: Prompt Inspector & System/User Prompts |
| **4** | **LangChain Core Components** | Built entirely using LangChain Prompt Templates (`ChatPromptTemplate`) and LangChain Expression Language (`LCEL` pipe syntax: `prompt \| llm \| StrOutputParser()`). | `core/intent_router.py`, `agents/*.py` |
| **5** | **Agentic RAG** | Intelligent router decides *when* to retrieve from ChromaDB, constructs targeted semantic search queries, retrieves top-k chunks, and attaches section/page citations. | `core/intent_router.py`, `agents/orchestrator.py` |
| **6** | **Intent Detection & Routing** | Analyzes natural language queries and dynamically classifies into: `SUMMARIZE`, `CONCEPT_EXTRACT`, `SLIDE_GENERATE`, `VIVA_QUESTIONS`, `INTERACTIVE_VIVA`, `PAPER_QA`. | Tab 1: Intent Badge & Routing Trace |
| **7** | **Vector Database (ChromaDB)** | ChromaDB persistent storage storing document embeddings (`all-MiniLM-L6-v2`), metadata filtering by academic section and page number, and similarity search. | `core/vector_store.py` & Sidebar Status |

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+ (Tested on Python 3.13)
- Ollama (Optional for local LLM inference; API providers also supported)

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/paper-presentation-viva-agent.git
cd paper-presentation-viva-agent
pip install -r requirements.txt
```

### 3. Local LLM Setup (Ollama)
Ollama is pre-configured to run locally on `http://localhost:11434`:
```bash
# Start Ollama service (if not running)
ollama serve

# Pull local model (e.g. lightweight 1.3 GB llama3.2:1b)
ollama pull llama3.2:1b
```

### 4. Running the Application
Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
AGENTIC-AI/
├── app.py                      # Main Streamlit Application
├── config.py                   # Central configuration & criteria registry
├── requirements.txt            # Python dependencies
├── test_system.py              # Automated integration & acceptance test suite
├── sample_papers/
│   └── attention_is_all_you_need_sample.txt # Seminal NeurIPS benchmark paper
├── core/
│   ├── document_loader.py      # PDF & text parser with section detection
│   ├── vector_store.py         # ChromaDB client & similarity retrieval
│   ├── llm_factory.py          # Provider factory (Ollama, Gemini, Groq, OpenAI)
│   ├── intent_router.py        # LCEL-based Intent Detection & Dynamic Routing
│   └── prompt_registry.py      # All prompt templates & engineering specifications
├── agents/
│   ├── summarizer_agent.py     # Sectioned RAG academic summary
│   ├── concept_agent.py        # Novelty, architecture & metrics extraction
│   ├── slide_agent.py          # Presentation deck generator + PPTX exporter
│   ├── viva_agent.py           # 4-tier defense questions + interactive examiner
│   └── orchestrator.py         # Master coordinator
├── ui/
│   ├── styles.py               # Custom CSS styling
│   └── components.py           # Presentation cards, viva flashcards, badges
└── exports/                    # Auto-generated PowerPoint (.pptx) presentations
```

---

## 🎓 Agent Features Walkthrough

### 1. Structured Summarizer Agent
- Produces a standardized 6-part academic summary:
  1. Executive Abstract & Core Thesis
  2. Problem Formulation & Prior Art Limitations
  3. Proposed Methodology & Mathematical Mechanics
  4. Empirical Results & SOTA Benchmarks
  5. Critical Limitations & Resource Bounds
  6. Academic Significance
- Features explicit **Chain of Thought (CoT)** reasoning trace in `<thinking>` blocks.

### 2. Concept-Extractor Agent
- Pinpoints core algorithmic innovations, loss functions, attention formulations, and training tricks.
- Summarizes benchmark datasets (e.g. WMT'14 English-German/French), BLEU gains, and speedups.

### 3. Slide-Generator Agent & PowerPoint (.pptx) Download
- Transforms research into clean 6-8 slide conference presentations.
- Each slide features:
  - Slide category & punchy title
  - High-impact bullet points
  - Highlighted **Key Takeaway**
  - **Speaker Notes Script** for oral delivery
- Generates and downloads real `.pptx` presentation files with 1 click!

### 4. Viva Voce Defense Simulator
- **Anticipated Questions Deck**: Questions categorized across 4 difficulty tiers:
  - *Tier 1: Fundamental Concept Questions*
  - *Tier 2: Methodological Deep-Dives*
  - *Tier 3: Skeptical & Critical Edge-Case Questions*
  - *Tier 4: Limitations, Assumptions & Future Directions*
  - Includes **Examiner Intent**, **Gold-Standard Model Answers**, and **Common Traps to Avoid**.
- **Interactive Mock Defense Mode**: Simulates a live oral defense where the examiner tests the candidate, grades their response (/10), highlights weaknesses, and asks follow-ups!
