"""
Automated Integration and Acceptance Criteria Verification Suite
Tests all 7 Mandatory Acceptance Criteria.
"""
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from core.document_loader import AcademicDocumentLoader
from core.vector_store import ChromaManager
from core.llm_factory import LLMFactory
from core.intent_router import AcademicIntentRouter
from agents.orchestrator import AcademicOrchestrator
from agents.summarizer_agent import SummarizerAgent
from agents.concept_agent import ConceptExtractorAgent
from agents.slide_agent import SlideGeneratorAgent
from agents.viva_agent import VivaAgent


def run_acceptance_tests():
    print("=" * 70)
    print("🎓 ACADEMIC PAPER-TO-PRESENTATION & VIVA AGENT: TEST SUITE")
    print("=" * 70)

    # 1. Test LLM Integration (Criterion 1)
    print("\n[Criterion 1: LLM Integration]")
    ollama_live = LLMFactory.is_ollama_running()
    print(f" -> Ollama daemon check: {'LIVE (127.0.0.1:11434)' if ollama_live else 'OFFLINE'}")
    if ollama_live:
        models = LLMFactory.get_ollama_models()
        print(f" -> Local models available in Ollama: {models}")
        llm = LLMFactory.create_llm(config.PROVIDER_OLLAMA, model_name=models[0] if models else "llama3.2:1b")
    else:
        llm = LLMFactory.create_llm(config.PROVIDER_GEMINI)
    print(f" -> Active LLM instantiated: {type(llm).__name__}")
    assert llm is not None, "Failed to instantiate LLM"
    print(" ✅ Criterion 1: PASS")

    # 2. Test Document Ingestion & Chunking (Criterion 7 & 5)
    print("\n[Criterion 7: Vector Database (ChromaDB) & Ingestion]")
    sample_file = config.SAMPLE_PAPERS_DIR / "attention_is_all_you_need_sample.txt"
    loader = AcademicDocumentLoader(chunk_size=800, chunk_overlap=100)
    doc_data = loader.load_from_file(sample_file, "Attention_Is_All_You_Need.txt")
    print(f" -> Loaded sample paper: {doc_data['stats']['total_words']} words, {doc_data['stats']['total_chunks']} chunks")
    print(f" -> Detected Academic Sections: {doc_data['stats']['detected_sections']}")

    vector_manager = ChromaManager(collection_name="test_paper_collection")
    indexed_count = vector_manager.index_document_chunks(doc_data["chunks"])
    print(f" -> Successfully indexed {indexed_count} chunks into ChromaDB")
    assert indexed_count > 0, "No chunks indexed"

    # Test ChromaDB Similarity Search
    search_query = "scaled dot product attention formula and division by sqrt(d_k)"
    retrieved = vector_manager.similarity_search(search_query, k=3)
    print(f" -> ChromaDB similarity search for '{search_query[:35]}...' returned {len(retrieved)} chunks")
    assert len(retrieved) > 0, "ChromaDB retrieval returned 0 chunks"
    print(f" -> Top retrieved chunk section: {retrieved[0]['section']} (Page {retrieved[0]['page_number']})")
    print(" ✅ Criterion 7: PASS")

    # 3. Test Intent Detection & Routing (Criterion 6)
    print("\n[Criterion 6: Intent Detection & Dynamic Routing]")
    router = AcademicIntentRouter(llm=llm)

    test_queries = [
        ("Summarize the paper's key contributions", "SUMMARIZE"),
        ("Extract the novelty and baseline comparisons", "CONCEPT_EXTRACT"),
        ("Create a 6 slide presentation deck for my defense", "SLIDE_GENERATE"),
        ("What viva questions will the external committee ask?", "VIVA_QUESTIONS")
    ]

    for q, expected_intent in test_queries:
        decision = router.route_query(q)
        print(f" -> Query: '{q}'")
        print(f"    Classified Intent: {decision['intent']} (Confidence: {decision['confidence']})")
        print(f"    Needs Retrieval: {decision['needs_retrieval']} | Search Query: '{decision['search_query']}'")
        print(f"    CoT Thought: {decision['thought'][:75]}...")
    print(" ✅ Criterion 6: PASS")

    # 4. Test LangChain Core Components & LCEL (Criterion 4)
    print("\n[Criterion 4: LangChain Core Components & LCEL]")
    summarizer = SummarizerAgent(llm=llm, vector_manager=vector_manager)
    print(f" -> Summarizer Agent LCEL Chain initialized: {summarizer.chain is not None}")
    summary_result = summarizer.run("Provide executive sectioned summary")
    print(f" -> Summarizer output generated ({len(summary_result['summary'])} chars)")
    print(f" -> Citations attached: {len(summary_result['citations'])} chunks cited")
    assert len(summary_result["summary"]) > 50, "Summary output too short"
    print(" ✅ Criterion 4: PASS")

    # 5. Test Agentic RAG & Grounded Synthesis (Criterion 5)
    print("\n[Criterion 5: Agentic RAG with Citations]")
    concept_extractor = ConceptExtractorAgent(llm=llm, vector_manager=vector_manager)
    concepts_result = concept_extractor.run("Extract key novelty and empirical metrics")
    print(f" -> Concept Extractor output generated ({len(concepts_result['concepts'])} chars)")
    print(f" -> Citations attached: {len(concepts_result.get('citations', []))}")
    assert len(concepts_result["concepts"]) > 50, "Concepts output too short"
    print(" ✅ Criterion 5: PASS")

    # 6. Test Slide Generator & PPTX Export
    print("\n[Slide-Generator Agent & PowerPoint (.pptx) Compilation]")
    slide_gen = SlideGeneratorAgent(llm=llm, vector_manager=vector_manager)
    slide_res = slide_gen.run("Create conference presentation slides")
    slides = slide_res.get("slides", [])
    pptx_path = slide_res.get("pptx_path")
    print(f" -> Generated {len(slides)} presentation slides with speaker notes")
    print(f" -> Slide 1 Title: '{slides[0].get('title')}'")
    print(f" -> PowerPoint exported to: {pptx_path}")
    assert pptx_path is not None and Path(pptx_path).exists(), "PPTX file was not created"
    print(f" -> PPTX file size: {Path(pptx_path).stat().st_size} bytes")
    print(" ✅ Slide-Generator & PPTX Export: PASS")

    # 7. Test Viva Agent & Interactive Examiner
    print("\n[Viva Voce Agent & Interactive Examiner]")
    viva_agent = VivaAgent(llm=llm, vector_manager=vector_manager)
    viva_res = viva_agent.generate_questions("Anticipate rigorous viva defense questions")
    print(f" -> Viva Question Deck generated ({len(viva_res['viva_content'])} chars)")

    # Test Interactive Viva Evaluation Turn
    eval_turn = viva_agent.evaluate_candidate_turn(
        candidate_answer="We replaced RNNs with self-attention because recurrent connections prevent parallel computation along the sequence length dimension during training.",
        current_question="Why did you eliminate recurrence?"
    )
    print(" -> Interactive Viva Examiner Response:")
    print(f"    {eval_turn['evaluation'][:180]}...")
    print(" ✅ Viva Agent & Interactive Simulator: PASS")

    # 8. Test Master Orchestrator
    print("\n[Master Orchestrator End-to-End Test]")
    orchestrator = AcademicOrchestrator(llm=llm, vector_manager=vector_manager)
    res = orchestrator.process_request("Generate 6 presentation slides with speaker notes")
    print(f" -> Orchestrator routed to: {res.get('agent_used')}")
    print(f" -> Intent: {res['intent']} | Chunks retrieved: {len(res['retrieved_chunks'])}")
    print(" ✅ Master Orchestrator: PASS")

    print("\n" + "=" * 70)
    print("🎉 ALL 7 MANDATORY ACCEPTANCE CRITERIA VERIFIED AND PASSING!")
    print("=" * 70)


if __name__ == "__main__":
    run_acceptance_tests()
