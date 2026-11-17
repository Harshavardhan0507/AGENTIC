"""
Intent Detection and Dynamic Routing Agent using LangChain LCEL
Demonstrates:
  - LCEL (Prompt | LLM | Parser)
  - Intent Classification (SUMMARIZE, CONCEPT_EXTRACT, SLIDE_GENERATE, VIVA_QUESTIONS, INTERACTIVE_VIVA, PAPER_QA)
  - Retrieval query generation & routing decision
"""
import json
import re
from typing import Dict, Any, Optional

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    ChatPromptTemplate = None
    StrOutputParser = None

from core.prompt_registry import INTENT_ROUTER_SYSTEM_PROMPT


class AcademicIntentRouter:
    """
    Analyzes user queries, determines intent, and routes to specialized agents
    using a LangChain LCEL chain.
    """
    def __init__(self, llm: Any):
        self.llm = llm
        self.chain = self._build_lcel_chain()

    def _build_lcel_chain(self):
        """Constructs LangChain Expression Language (LCEL) pipeline."""
        if not ChatPromptTemplate or not StrOutputParser:
            return None

        prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_ROUTER_SYSTEM_PROMPT),
            ("human", "User Query: {query}\nConversation History: {history}")
        ])

        # LCEL Pipe: prompt | llm | StrOutputParser()
        return prompt | self.llm | StrOutputParser()

    def route_query(self, query: str, history: str = "") -> Dict[str, Any]:
        """
        Executes intent detection and returns routing metadata.
        """
        # Try LCEL execution
        raw_output = ""
        if self.chain:
            try:
                raw_output = self.chain.invoke({"query": query, "history": history})
            except Exception as e:
                print(f"LCEL Intent Router execution exception: {e}")
                raw_output = ""

        # Fallback to direct prompt synthesis if chain returned empty
        if not raw_output and hasattr(self.llm, "_synthesize_response"):
            raw_output = self.llm._synthesize_response(INTENT_ROUTER_SYSTEM_PROMPT, query)

        # Parse JSON output from LLM
        routing_decision = self._parse_routing_output(raw_output, query)
        return routing_decision

    def _parse_routing_output(self, raw_output: str, original_query: str) -> Dict[str, Any]:
        """Parses structured JSON routing object with resilient heuristics."""
        clean_json_str = raw_output.strip()

        # Extract markdown json block if wrapped
        if "```json" in clean_json_str:
            clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json_str:
            clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()

        # Find first '{' and last '}'
        start_idx = clean_json_str.find("{")
        end_idx = clean_json_str.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_json_str = clean_json_str[start_idx:end_idx + 1]

        try:
            data = json.loads(clean_json_str)
            intent = data.get("intent", "").upper()
            valid_intents = [
                "SUMMARIZE", "CONCEPT_EXTRACT", "SLIDE_GENERATE",
                "VIVA_QUESTIONS", "INTERACTIVE_VIVA", "PAPER_QA", "GENERAL_CONVERSATION"
            ]
            if intent not in valid_intents:
                intent = self._heuristic_intent(original_query)

            return {
                "intent": intent,
                "thought": data.get("thought", "Classified based on contextual query features."),
                "confidence": float(data.get("confidence", 0.9)),
                "needs_retrieval": bool(data.get("needs_retrieval", True)),
                "search_query": str(data.get("search_query", original_query)),
                "raw_output": raw_output
            }
        except Exception:
            # Resilient rule-based fallback
            intent = self._heuristic_intent(original_query)
            return {
                "intent": intent,
                "thought": f"Heuristic classification selected {intent} from user keywords.",
                "confidence": 0.85,
                "needs_retrieval": intent != "GENERAL_CONVERSATION",
                "search_query": original_query,
                "raw_output": raw_output
            }

    def _heuristic_intent(self, text: str) -> str:
        """Rule-based heuristic for intent classification."""
        t = text.lower()
        if any(w in t for w in ["slide", "presentation", "deck", "powerpoint", "pptx"]):
            return "SLIDE_GENERATE"
        elif any(w in t for w in ["viva", "defense", "examiner", "oral exam", "mock defense", "test me", "grill me"]):
            if any(w in t for w in ["test me", "grill me", "simulate", "ask me"]):
                return "INTERACTIVE_VIVA"
            return "VIVA_QUESTIONS"
        elif any(w in t for w in ["concept", "novelty", "contribution", "baseline", "architecture", "extract"]):
            return "CONCEPT_EXTRACT"
        elif any(w in t for w in ["summar", "overview", "tldr", "abstract", "brief", "digest"]):
            return "SUMMARIZE"
        elif any(w in t for w in ["hello", "hi", "help", "who are you", "what can you do"]):
            return "GENERAL_CONVERSATION"
        else:
            return "PAPER_QA"
