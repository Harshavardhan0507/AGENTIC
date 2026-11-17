"""
Document Loader and Preprocessor for Academic Papers
Supports:
  - PDF files (.pdf) using pdfplumber / pypdf
  - Text & Markdown papers (.txt, .md)
  - Section extraction (Abstract, Intro, Architecture, Results, Limitations, etc.)
  - Semantic chunking with rich metadata for ChromaDB RAG
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback if package is still installing
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=150, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", ". ", " "]

        def split_text(self, text: str) -> List[str]:
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                start += self.chunk_size - self.chunk_overlap
            return chunks

# Standard academic paper section patterns
SECTION_PATTERNS = [
    (r"(?i)\b(abstract)\b", "Abstract"),
    (r"(?i)\b(1\.?\s*introduction|background)\b", "Introduction"),
    (r"(?i)\b(2\.?\s*related\s+work|prior\s+art)\b", "Related Work"),
    (r"(?i)\b(3\.?\s*(?:model|architecture|methodology|proposed\s+approach|methods?))\b", "Methodology & Architecture"),
    (r"(?i)\b(4\.?\s*(?:experiments?|results?|empirical\s+evaluation|benchmarks?))\b", "Experiments & Results"),
    (r"(?i)\b(5\.?\s*(?:discussion|limitations?|ablations?))\b", "Discussion & Limitations"),
    (r"(?i)\b(6\.?\s*(?:conclusions?|future\s+work))\b", "Conclusion & Future Work"),
    (r"(?i)\b(references|bibliography)\b", "References"),
]


class DocumentChunk:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"DocumentChunk(chunk_id={self.metadata.get('chunk_id')}, section={self.metadata.get('section')}, len={len(self.page_content)})"


class AcademicDocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        )

    def load_from_file(self, file_path_or_buffer, file_name: str) -> Dict[str, Any]:
        """
        Loads document from path or uploaded buffer, extracts text and section metadata,
        and splits into chunks.
        """
        file_name_lower = file_name.lower()

        if file_name_lower.endswith(".pdf"):
            full_text, pages_data = self._extract_pdf(file_path_or_buffer)
        else:
            full_text, pages_data = self._extract_text(file_path_or_buffer)

        # Build chunks with section awareness
        chunks = self._chunk_document(full_text, pages_data, file_name)

        # Detect sections present
        detected_sections = list(set(c.metadata["section"] for c in chunks if c.metadata.get("section") != "General"))

        stats = {
            "file_name": file_name,
            "total_chars": len(full_text),
            "total_words": len(full_text.split()),
            "total_pages": len(pages_data),
            "total_chunks": len(chunks),
            "detected_sections": sorted(detected_sections),
            "full_text": full_text
        }

        return {
            "stats": stats,
            "chunks": chunks
        }

    def _extract_pdf(self, file_or_buffer) -> tuple[str, List[Dict[str, Any]]]:
        """Extracts text per page from PDF using pdfplumber or pypdf."""
        pages_data = []
        full_text_parts = []

        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(file_or_buffer) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    full_text_parts.append(text)
                    pages_data.append({
                        "page_number": page_idx + 1,
                        "text": text,
                        "char_count": len(text)
                    })
            return "\n\n".join(full_text_parts), pages_data
        except Exception:
            pass

        # Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(file_or_buffer)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                full_text_parts.append(text)
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": text,
                    "char_count": len(text)
                })
            return "\n\n".join(full_text_parts), pages_data
        except Exception as e:
            raise RuntimeError(f"Could not parse PDF: {str(e)}")

    def _extract_text(self, file_or_buffer) -> tuple[str, List[Dict[str, Any]]]:
        """Extracts text from plain text or markdown files."""
        if hasattr(file_or_buffer, "read"):
            content = file_or_buffer.read()
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="ignore")
            else:
                text = str(content)
        elif isinstance(file_or_buffer, (str, Path)):
            with open(file_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            text = str(file_or_buffer)

        # Estimate virtual pages (approx 3000 chars per academic page)
        page_size = 3000
        pages_data = []
        start = 0
        page_idx = 1
        while start < len(text):
            chunk = text[start:start + page_size]
            pages_data.append({
                "page_number": page_idx,
                "text": chunk,
                "char_count": len(chunk)
            })
            start += page_size
            page_idx += 1

        return text, pages_data

    def _detect_section(self, text: str, current_section: str = "General") -> str:
        """Determines academic section heading from text."""
        for pattern, section_name in SECTION_PATTERNS:
            if re.search(pattern, text):
                return section_name
        return current_section

    def _chunk_document(self, full_text: str, pages_data: List[Dict[str, Any]], file_name: str) -> List[DocumentChunk]:
        """Splits full text into chunks while preserving section and page metadata."""
        raw_chunks = self.text_splitter.split_text(full_text)
        doc_chunks = []
        current_section = "General"

        for idx, chunk_text in enumerate(raw_chunks):
            current_section = self._detect_section(chunk_text[:200], current_section)

            # Estimate page number based on offset
            page_num = 1
            if pages_data:
                # Find matching page
                for p in pages_data:
                    if chunk_text[:50] in p["text"]:
                        page_num = p["page_number"]
                        break

            metadata = {
                "chunk_id": idx,
                "source": file_name,
                "section": current_section,
                "page_number": page_num,
                "char_length": len(chunk_text),
                "word_count": len(chunk_text.split())
            }

            doc_chunks.append(DocumentChunk(page_content=chunk_text, metadata=metadata))

        return doc_chunks
