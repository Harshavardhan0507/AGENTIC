"""
Configuration settings for Paper-to-Presentation & Viva Agent
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PAPERS_DIR = BASE_DIR / "sample_papers"
CHROMA_PERSIST_DIR = BASE_DIR / ".chroma_db"
EXPORTS_DIR = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# App Metadata
APP_NAME = "Academic Paper-to-Presentation & Viva Agent"
APP_SUBTITLE = "Agentic RAG Assistant for Academic Research, Slides Generation & Viva Defense Preparation"
VERSION = "1.0.0"

# RAG & Chunking Defaults
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
CHROMA_COLLECTION_NAME = "research_paper_collection"

# LLM Providers & Supported Models
PROVIDER_OLLAMA = "Ollama (Local)"
PROVIDER_GEMINI = "Google Gemini (API)"
PROVIDER_GROQ = "Groq (API - Fast)"
PROVIDER_OPENAI = "OpenAI (API)"

SUPPORTED_PROVIDERS = [
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA,
    PROVIDER_GROQ,
    PROVIDER_OPENAI,
]

DEFAULT_MODELS = {
    PROVIDER_OLLAMA: ["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b", "qwen2.5:1.5b", "mistral:latest", "deepseek-r1:1.5b"],
    PROVIDER_GEMINI: ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
    PROVIDER_GROQ: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    PROVIDER_OPENAI: ["gpt-4o-mini", "gpt-4o"],
}

# Mandatory Acceptance Criteria Checklist
ACCEPTANCE_CRITERIA = [
    {
        "id": 1,
        "name": "LLM Integration",
        "description": "Demonstrates local LLM (Ollama) and API-based models (Gemini, Groq, OpenAI).",
        "status": "Demonstrated"
    },
    {
        "id": 2,
        "name": "Interactive GUI",
        "description": "Streamlit GUI for paper upload, structured summary, concept extraction, slide preview/export, and viva practice.",
        "status": "Demonstrated"
    },
    {
        "id": 3,
        "name": "Prompt Engineering",
        "description": "Demonstrates Chain of Thought (CoT), Few-Shot prompting, ReAct retrieval, and Reflection critique.",
        "status": "Demonstrated"
    },
    {
        "id": 4,
        "name": "LangChain Core Components",
        "description": "Uses LangChain Prompt Templates (ChatPromptTemplate) and LCEL (| pipe syntax) to connect prompts, models, and parsers.",
        "status": "Demonstrated"
    },
    {
        "id": 5,
        "name": "Agentic RAG",
        "description": "Agent decides when to query ChromaDB, retrieves section-specific context, and produces grounded answers with citations.",
        "status": "Demonstrated"
    },
    {
        "id": 6,
        "name": "Intent Detection & Routing",
        "description": "Classifies user intent (Summary, Concept Extraction, Slides, Viva, QA) and routes to specialized agents.",
        "status": "Demonstrated"
    },
    {
        "id": 7,
        "name": "Vector Database (ChromaDB)",
        "description": "ChromaDB stores document embeddings and performs similarity search with metadata filtering.",
        "status": "Demonstrated"
    }
]
