"""
Slide-Generator Agent
Structures research content into conference/defense presentation slides,
and exports presentation decks to PowerPoint (.pptx).
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    ChatPromptTemplate = None
    StrOutputParser = None

from core.prompt_registry import SLIDE_GENERATOR_SYSTEM_PROMPT, SLIDE_GENERATOR_USER_PROMPT
from core.vector_store import ChromaManager
import config


class SlideGeneratorAgent:
    """
    Transforms research papers into structured presentation slide decks
    with speaker notes, and generates native PowerPoint (.pptx) files.
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
            ("system", SLIDE_GENERATOR_SYSTEM_PROMPT),
            ("human", SLIDE_GENERATOR_USER_PROMPT)
        ])

        return prompt | self.llm | StrOutputParser()

    def run(self, query: str = "Generate a 6-slide academic presentation deck", top_k: int = 6) -> Dict[str, Any]:
        """
        Retrieves context and generates structured slide deck JSON.
        """
        search_query = query if len(query.strip()) > 5 else "abstract motivation architecture experimental results limitations conclusion"
        retrieved_chunks = self.vector_manager.similarity_search(search_query, k=top_k)

        context_str = "\n\n---\n\n".join([f"[{c['section']} P.{c['page_number']}]:\n{c['content']}" for c in retrieved_chunks])
        if not context_str:
            context_str = "No vector chunks found. Use available knowledge."

        raw_output = ""
        if self.chain:
            try:
                raw_output = self.chain.invoke({
                    "context": context_str,
                    "query": query
                })
            except Exception as e:
                print(f"SlideGeneratorAgent LCEL error: {e}")
                raw_output = ""

        if not raw_output and hasattr(self.llm, "_synthesize_response"):
            user_msg = SLIDE_GENERATOR_USER_PROMPT.format(context=context_str, query=query)
            raw_output = self.llm._synthesize_response(SLIDE_GENERATOR_SYSTEM_PROMPT, user_msg)

        slides = self._parse_slides(raw_output)

        # Generate PowerPoint file
        pptx_path = self.export_to_pptx(slides, paper_title="Academic Presentation")

        return {
            "slides": slides,
            "pptx_path": str(pptx_path) if pptx_path else None,
            "raw_output": raw_output,
            "agent": "Slide-Generator Agent"
        }

    def _parse_slides(self, text: str) -> List[Dict[str, Any]]:
        """Parses JSON slide array from LLM output."""
        clean = text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        start = clean.find("[")
        end = clean.rfind("]")
        if start != -1 and end != -1:
            clean = clean[start:end + 1]

        try:
            slides = json.loads(clean)
            if isinstance(slides, list) and len(slides) > 0:
                return slides
        except Exception:
            pass

        # Robust fallback default slides if JSON parsing failed
        return [
            {
                "slide_number": 1,
                "title": "Research Paper Overview",
                "subtitle": "Paper-to-Presentation Agent Summary",
                "bullet_points": [
                    "Synthesizes key architectural contributions",
                    "Breaks down experimental benchmarks and baseline comparisons",
                    "Prepares candidates with deep speaker notes for academic defense"
                ],
                "key_takeaway": "Automated academic presentation structuring from grounded ChromaDB context.",
                "speaker_notes": "Welcome esteemed audience. In this presentation, we outline the primary contributions and empirical validations presented in the paper."
            }
        ]

    def export_to_pptx(self, slides: List[Dict[str, Any]], paper_title: str = "Research Presentation") -> Optional[Path]:
        """
        Creates a PowerPoint (.pptx) file using python-pptx with styled typography,
        content cards, and speaker notes.
        """
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            from pptx.dml.color import RGBColor
            from pptx.enum.text import PP_ALIGN

            prs = Presentation()
            # 16:9 widescreen dimensions
            prs.slide_width = Inches(13.333)
            prs.slide_height = Inches(7.5)
            blank_layout = prs.slide_layouts[6]

            # Color Palette
            NAVY = RGBColor(15, 23, 42)        # #0f172a
            SLATE = RGBColor(71, 85, 105)      # #475569
            BLUE = RGBColor(37, 99, 235)       # #2563eb
            LIGHT_BG = RGBColor(248, 250, 252) # #f8fafc
            WHITE = RGBColor(255, 255, 255)
            AMBER = RGBColor(217, 119, 6)      # #d97706

            for s in slides:
                slide = prs.slides.add_slide(blank_layout)

                # Slide Header Box
                header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.7), Inches(1.2))
                tf = header_box.text_frame
                tf.word_wrap = True

                p_num = tf.paragraphs[0]
                p_num.text = f"SLIDE {s.get('slide_number', 1)}  •  {s.get('subtitle', 'RESEARCH OVERVIEW').upper()}"
                p_num.font.size = Pt(11)
                p_num.font.bold = True
                p_num.font.color.rgb = BLUE

                p_title = tf.add_paragraph()
                p_title.text = s.get("title", "Untitled Slide")
                p_title.font.size = Pt(24)
                p_title.font.bold = True
                p_title.font.color.rgb = NAVY

                # Main Content Box (Bullet points)
                content_box = slide.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(3.6))
                c_tf = content_box.text_frame
                c_tf.word_wrap = True

                bullets = s.get("bullet_points", [])
                for b_idx, bullet in enumerate(bullets):
                    p = c_tf.paragraphs[0] if b_idx == 0 else c_tf.add_paragraph()
                    p.text = f"•  {bullet}"
                    p.font.size = Pt(16)
                    p.font.color.rgb = NAVY
                    p.space_after = Pt(14)

                # Key Takeaway Box
                takeaway = s.get("key_takeaway")
                if takeaway:
                    takeaway_box = slide.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(11.7), Inches(0.9))
                    t_tf = takeaway_box.text_frame
                    t_tf.word_wrap = True
                    p_t = t_tf.paragraphs[0]
                    p_t.text = f"💡 KEY TAKEAWAY: {takeaway}"
                    p_t.font.size = Pt(13)
                    p_t.font.bold = True
                    p_t.font.color.rgb = AMBER

                # Embed Speaker Notes into PowerPoint speaker note pane
                speaker_notes = s.get("speaker_notes", "")
                if speaker_notes:
                    notes_slide = slide.notes_slide
                    notes_tf = notes_slide.notes_text_frame
                    notes_tf.text = speaker_notes

            # Save file
            safe_filename = "presentation_deck.pptx"
            output_file = config.EXPORTS_DIR / safe_filename
            prs.save(str(output_file))
            return output_file

        except Exception as e:
            print(f"PPTX Export error: {e}")
            return None
