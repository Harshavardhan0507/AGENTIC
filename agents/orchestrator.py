"""
Master Multi-Agent Orchestrator
Coordinates Intent Routing, Agentic RAG Retrieval, and Specialized Sub-Agents.
Maintains conversational state and execution tracing.
"""
from typing import Dict, Any, List, Optional
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    ChatPromptTemplate = None
    StrOutputParser = None

from core.intent_router import AcademicIntentRouter
from core.vector_store import ChromaManager
from agents.summarizer_agent import SummarizerAgent
from agents.concept_agent import ConceptExtractorAgent
from agents.slide_agent import SlideGeneratorAgent
from agents.viva_agent import VivaAgent


class AcademicOrchestrator:
    """
    Main orchestrator implementing Agentic RAG and Intent Routing.
    """
    def __init__(self, llm: Any, vector_manager: ChromaManager):
        self.llm = llm
        self.vector_manager = vector_manager
        self.router = AcademicIntentRouter(llm=self.llm)
        self.summarizer = SummarizerAgent(llm=self.llm, vector_manager=self.vector_manager)
        self.concept_extractor = ConceptExtractorAgent(llm=self.llm, vector_manager=self.vector_manager)
        self.slide_generator = SlideGeneratorAgent(llm=self.llm, vector_manager=self.vector_manager)
        self.viva_agent = VivaAgent(llm=self.llm, vector_manager=self.vector_manager)

    def process_request(self, user_query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        End-to-end request handling:
          1. Intent Detection via LCEL
          2. Decision on whether to retrieve from ChromaDB
          3. Execution by targeted agent
          4. Grounded response assembly
        """
        history = history or []
        formatted_history = "\n".join([f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-4:]])

        # Step 1: Detect Intent
        routing = self.router.route_query(user_query, formatted_history)
        intent = routing["intent"]
        needs_retrieval = routing["needs_retrieval"]
        search_query = routing["search_query"]

        retrieved_chunks = []
        if needs_retrieval:
            retrieved_chunks = self.vector_manager.similarity_search(search_query, k=4)

        result_payload = {
            "query": user_query,
            "intent": intent,
            "routing_thought": routing["thought"],
            "confidence": routing["confidence"],
            "needs_retrieval": needs_retrieval,
            "search_query": search_query,
            "retrieved_chunks": retrieved_chunks,
            "response": "",
            "extra_data": {}
        }

        # Step 2: Route to Specialized Agent
        if intent == "SUMMARIZE":
            summary_res = self.summarizer.run(user_query)
            result_payload["response"] = summary_res["summary"]
            result_payload["extra_data"]["citations"] = summary_res.get("citations", [])
            result_payload["extra_data"]["cot_reasoning"] = summary_res.get("cot_reasoning", "")
            result_payload["agent_used"] = "Summarizer Agent"

        elif intent == "CONCEPT_EXTRACT":
            concept_res = self.concept_extractor.run(user_query)
            result_payload["response"] = concept_res["concepts"]
            result_payload["extra_data"]["citations"] = concept_res.get("citations", [])
            result_payload["agent_used"] = "Concept-Extractor Agent"

        elif intent == "SLIDE_GENERATE":
            slide_res = self.slide_generator.run(user_query)
            result_payload["response"] = f"Generated a presentation deck with {len(slide_res.get('slides', []))} slides."
            result_payload["extra_data"]["slides"] = slide_res.get("slides", [])
            result_payload["extra_data"]["pptx_path"] = slide_res.get("pptx_path")
            result_payload["agent_used"] = "Slide-Generator Agent"

        elif intent == "VIVA_QUESTIONS":
            viva_res = self.viva_agent.generate_questions(user_query)
            result_payload["response"] = viva_res["viva_content"]
            result_payload["agent_used"] = "Viva Agent"

        elif intent == "INTERACTIVE_VIVA":
            # If user provided an answer in context, evaluate it; otherwise pose starting question
            eval_res = self.viva_agent.evaluate_candidate_turn(user_query)
            result_payload["response"] = eval_res["evaluation"]
            result_payload["agent_used"] = "Viva Agent (Interactive Examiner)"

        elif intent == "GENERAL_CONVERSATION":
            result_payload["response"] = (
                "Hello! I am your Academic Paper-to-Presentation and Viva Agent. "
                "You can ask me to summarize an uploaded paper, extract key novel concepts, "
                "generate presentation slides with speaker notes (and download as .pptx), "
                "or practice viva voce defense questions."
            )
            result_payload["agent_used"] = "Conversational Handler"

        else: # PAPER_QA
            context_text = "\n\n".join([f"[{c['section']} P.{c['page_number']}]: {c['content']}" for c in retrieved_chunks])
            result_payload["response"] = self._answer_qa(user_query, context_text)
            result_payload["agent_used"] = "Paper-Grounded QA Agent"

        return result_payload

    def _answer_qa(self, query: str, context: str) -> str:
        """Grounded Paper QA answer generation using LCEL."""
        if not ChatPromptTemplate or not StrOutputParser:
            return f"Answer grounded in paper context:\n\n{context[:400]}..."

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an Academic Research Assistant. Answer the question precisely based ONLY on the provided context. Cite the section and page number when answering. If the context does not contain the answer, explicitly state that."),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"context": context, "query": query})
        except Exception:
            return f"Based on the paper excerpts:\n\n{context[:400]}"
