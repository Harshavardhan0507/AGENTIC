"""
Summarizer Agent
Produces a concise, sectioned academic summary grounded in ChromaDB retrieval.
Uses LangChain LCEL and Chain of Thought (CoT) prompting.
"""
from typing import Dict, Any, List, Optional
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    ChatPromptTemplate = None
    StrOutputParser = None

from core.prompt_registry import SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_PROMPT
from core.vector_store import ChromaManager


class SummarizerAgent:
    """
    Synthesizes structured, sectioned academic summaries from ChromaDB paper chunks.
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
            ("system", SUMMARIZER_SYSTEM_PROMPT),
            ("human", SUMMARIZER_USER_PROMPT)
        ])

        # LCEL: prompt | llm | StrOutputParser()
        return prompt | self.llm | StrOutputParser()

    def run(self, query: str = "Provide a comprehensive academic summary", top_k: int = 5) -> Dict[str, Any]:
        """
        Executes grounded RAG retrieval and synthesis.
        """
        # 1. RAG Retrieval from ChromaDB
        search_query = query if len(query.strip()) > 5 else "abstract introduction methodology architecture results conclusion"
        retrieved_chunks = self.vector_manager.similarity_search(search_query, k=top_k)

        # Format context with citations
        context_blocks = []
        citations = []
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            sec = meta.get("section", "General")
            page = meta.get("page_number", 1)
            cid = meta.get("chunk_id", 0)
            score = c.get("score", 0.0)
            citation_tag = f"[Section: {sec} | Page {page} | Chunk #{cid}]"
            citations.append({
                "citation": citation_tag,
                "section": sec,
                "page": page,
                "chunk_id": cid,
                "score": score,
                "excerpt": c.get("content", "")[:180] + "..."
            })
            context_blocks.append(f"{citation_tag}\n{c.get('content', '')}")

        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found in vector database."

        # 2. LCEL Execution
        summary_text = ""
        if self.chain:
            try:
                summary_text = self.chain.invoke({
                    "context": context_str,
                    "query": query
                })
            except Exception as e:
                print(f"SummarizerAgent LCEL error: {e}")
                summary_text = ""

        # Fallback if mock model or direct invoke
        if not summary_text and hasattr(self.llm, "_synthesize_response"):
            user_msg = SUMMARIZER_USER_PROMPT.format(context=context_str, query=query)
            summary_text = self.llm._synthesize_response(SUMMARIZER_SYSTEM_PROMPT, user_msg)

        # Separate <thinking> CoT block if present
        thinking_cot = ""
        final_summary = summary_text
        if "<thinking>" in summary_text and "</thinking>" in summary_text:
            parts = summary_text.split("</thinking>")
            thinking_cot = parts[0].replace("<thinking>", "").strip()
            final_summary = parts[1].strip()

        return {
            "summary": final_summary,
            "cot_reasoning": thinking_cot,
            "citations": citations,
            "raw_context": context_str,
            "agent": "Summarizer Agent"
        }
