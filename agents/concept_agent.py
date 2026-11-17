"""
Concept-Extractor Agent
Identifies key contributions, methods, and results.
Uses LangChain LCEL with structured technical taxonomy.
"""
from typing import Dict, Any, List, Optional
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    ChatPromptTemplate = None
    StrOutputParser = None

from core.prompt_registry import CONCEPT_EXTRACTOR_SYSTEM_PROMPT, CONCEPT_EXTRACTOR_USER_PROMPT
from core.vector_store import ChromaManager


class ConceptExtractorAgent:
    """
    Extracts deep conceptual mechanics, novel claims, empirical results, and limitations.
    """
    def __init__(self, llm: Any, vector_manager: ChromaManager):
        self.llm = llm
        self.vector_manager = vector_manager
        self.chain = self._build_lcel_chain()

    def _build_lcel_chain(self):
        """Constructs LangChain Expression Language (LCEL) chain."""
        if not ChatPromptTemplate or not StrOutputParser:
            return None

        prompt = ChatPromptTemplate.from_messages([
            ("system", CONCEPT_EXTRACTOR_SYSTEM_PROMPT),
            ("human", CONCEPT_EXTRACTOR_USER_PROMPT)
        ])

        return prompt | self.llm | StrOutputParser()

    def run(self, query: str = "Extract key contributions, methods, and results", top_k: int = 5) -> Dict[str, Any]:
        """
        Executes concept extraction using grounded ChromaDB chunks.
        """
        search_query = query if len(query.strip()) > 5 else "novelty contributions proposed architecture experimental results baseline"
        retrieved_chunks = self.vector_manager.similarity_search(search_query, k=top_k)

        # Context assembly
        context_blocks = []
        citations = []
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            sec = meta.get("section", "General")
            page = meta.get("page_number", 1)
            cid = meta.get("chunk_id", 0)
            score = c.get("score", 0.0)
            tag = f"[{sec} | P.{page} | Chunk #{cid}]"
            citations.append({
                "citation": tag,
                "section": sec,
                "page": page,
                "chunk_id": cid,
                "score": score,
                "excerpt": c.get("content", "")[:180] + "..."
            })
            context_blocks.append(f"{tag}\n{c.get('content', '')}")

        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found in vector database."

        # LCEL execution
        extraction_text = ""
        if self.chain:
            try:
                extraction_text = self.chain.invoke({
                    "context": context_str,
                    "query": query
                })
            except Exception as e:
                print(f"ConceptExtractorAgent LCEL error: {e}")
                extraction_text = ""

        if not extraction_text and hasattr(self.llm, "_synthesize_response"):
            user_msg = CONCEPT_EXTRACTOR_USER_PROMPT.format(context=context_str, query=query)
            extraction_text = self.llm._synthesize_response(CONCEPT_EXTRACTOR_SYSTEM_PROMPT, user_msg)

        return {
            "concepts": extraction_text,
            "citations": citations,
            "agent": "Concept-Extractor Agent"
        }
