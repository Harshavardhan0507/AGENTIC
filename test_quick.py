"""
Quick sanity check for LCEL prompt templates and chains.
"""
import sys
from pathlib import Path

# UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from core.llm_factory import LLMFactory
from core.intent_router import AcademicIntentRouter
from core.vector_store import ChromaManager
from agents.slide_agent import SlideGeneratorAgent

print("Testing LCEL chains with prompt template escaping...")

# Use mock model to test prompt template variable resolution in LCEL instantly
mock_llm = LLMFactory.create_llm(provider="MOCK_TEST")
router = AcademicIntentRouter(llm=mock_llm)

print(f"Intent Router LCEL Chain exists: {router.chain is not None}")
decision = router.route_query("Summarize this paper for me")
print(f"Intent detected: {decision['intent']}, Needs Retrieval: {decision['needs_retrieval']}")

vector_manager = ChromaManager(collection_name="quick_test_col")
slide_gen = SlideGeneratorAgent(llm=mock_llm, vector_manager=vector_manager)
print(f"Slide Generator LCEL Chain exists: {slide_gen.chain is not None}")
slide_res = slide_gen.run("Create 6 presentation slides")
print(f"Slides generated: {len(slide_res.get('slides', []))}")
print(f"Slide 1: {slide_res['slides'][0]['title']}")

print("\nAll LCEL chains executed with ZERO prompt errors!")
