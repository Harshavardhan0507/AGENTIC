"""
Central LLM Factory supporting Local (Ollama) and API Providers (Gemini, Groq, OpenAI).
Integrates seamlessly with LangChain LCEL pipelines.
"""
import os
from typing import Optional, Any, List, Dict
import urllib.request
import json

from pydantic import Field
import config

# Fallback base class in case langchain_core is still completing installation
try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, AIMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
except ImportError:
    BaseChatModel = object
    BaseMessage = object
    AIMessage = object
    ChatResult = object
    ChatGeneration = object


class MockAcademicChatModel(BaseChatModel):
    """
    Simulation / Academic Fallback Chat Model.
    Inherits from LangChain BaseChatModel so it executes inside LCEL pipelines
    (prompt | model | parser) even when no external API key is configured.
    """
    model_name: str = "Academic-Simulation-Engine"

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        # Extract last user query or context
        last_msg = messages[-1].content if messages else ""
        system_msg = messages[0].content if len(messages) > 1 else ""

        response_text = self._synthesize_response(system_msg, str(last_msg))
        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])

    def _synthesize_response(self, system_prompt: str, user_prompt: str) -> str:
        prompt_lower = (system_prompt + " " + user_prompt).lower()

        # 1. Intent Detection
        if "academic intent router" in prompt_lower or "needs_retrieval" in prompt_lower:
            if "slide" in user_prompt.lower() or "presentation" in user_prompt.lower():
                return json.dumps({
                    "thought": "User is asking for slide generation for a seminar or presentation. Needs paper-wide context.",
                    "intent": "SLIDE_GENERATE",
                    "confidence": 0.98,
                    "needs_retrieval": True,
                    "search_query": "abstract methodology results architecture conclusion"
                })
            elif "viva" in user_prompt.lower() or "defense" in user_prompt.lower() or "question" in user_prompt.lower():
                return json.dumps({
                    "thought": "User is seeking viva examination questions and model answers.",
                    "intent": "VIVA_QUESTIONS",
                    "confidence": 0.97,
                    "needs_retrieval": True,
                    "search_query": "limitations assumptions complexity failure modes"
                })
            elif "concept" in user_prompt.lower() or "novelty" in user_prompt.lower() or "contribution" in user_prompt.lower():
                return json.dumps({
                    "thought": "User wants key concepts, novelty, and theoretical contributions.",
                    "intent": "CONCEPT_EXTRACT",
                    "confidence": 0.96,
                    "needs_retrieval": True,
                    "search_query": "novelty contributions architecture experimental comparison"
                })
            else:
                return json.dumps({
                    "thought": "User is requesting a structured overview or summary of the paper.",
                    "intent": "SUMMARIZE",
                    "confidence": 0.95,
                    "needs_retrieval": True,
                    "search_query": "abstract introduction architecture results conclusion"
                })

        # 2. Slide Deck Generation
        if "slide-generator" in prompt_lower or "slide_number" in prompt_lower:
            return json.dumps([
                {
                    "slide_number": 1,
                    "title": "Attention Is All You Need",
                    "subtitle": "A Pure Attention Paradigm for Neural Transduction",
                    "bullet_points": [
                        "Eliminates sequential recurrence (RNNs/LSTMs) and convolutions",
                        "Introduces the Transformer: stacked multi-head self-attention",
                        "Enables massive parallelization across token sequences during training",
                        "Sets new State-of-the-Art in machine translation benchmarks"
                    ],
                    "key_takeaway": "Self-attention provides constant computational path length between any two tokens.",
                    "speaker_notes": "Esteemed examiners, today I present the Transformer architecture. Prior sequence transduction relied on recurrent steps that blocked parallel processing. This paper demonstrates that attention mechanisms alone are sufficient."
                },
                {
                    "slide_number": 2,
                    "title": "The Sequential Bottleneck",
                    "subtitle": "Motivation & Problem Formulation",
                    "bullet_points": [
                        "RNNs/LSTMs compute hidden states h_t sequentially from h_(t-1)",
                        "Precludes parallelization across training examples and sequence length",
                        "Long-distance dependencies degrade due to gradient vanishing/exploding",
                        "Memory constraints strictly limit batching capabilities"
                    ],
                    "key_takeaway": "Sequential dependencies impose a fundamental hardware underutilization barrier on modern GPUs.",
                    "speaker_notes": "Here we highlight the core friction: recurrent networks cannot compute position t until position t-1 is complete. This fundamentally bottlenecks parallel hardware."
                },
                {
                    "slide_number": 3,
                    "title": "Model Architecture",
                    "subtitle": "Encoder-Decoder Stacks & Sub-Layers",
                    "bullet_points": [
                        "6 identical Encoder layers and 6 Decoder layers (d_model = 512)",
                        "Each layer features Multi-Head Attention and Position-Wise Feed-Forward networks",
                        "Residual connections and Layer Normalization: LayerNorm(x + Sublayer(x))",
                        "Decoder applies causal masking to enforce auto-regressive property"
                    ],
                    "key_takeaway": "Residual connections and dimensional normalization ensure robust gradient propagation across 6 stacked layers.",
                    "speaker_notes": "The architecture maintains a consistent dimension of 512 throughout all sub-layers, allowing standard residual additions."
                },
                {
                    "slide_number": 4,
                    "title": "Scaled Dot-Product & Multi-Head Attention",
                    "subtitle": "Mathematical Mechanics",
                    "bullet_points": [
                        "Scaled Dot-Product: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
                        "Division by sqrt(d_k) counteracts extreme magnitude softmax saturation",
                        "Multi-Head Attention: Projects Q, K, V into h = 8 distinct subspaces (d_k = 64)",
                        "Allows model to simultaneously attend to information from different representation subspaces"
                    ],
                    "key_takeaway": "Multi-head attention prevents single-head averaging and captures diverse syntactic and semantic relations.",
                    "speaker_notes": "Notice the scaling factor 1 over square root of d_k. Without this scaling, dot products grow large for high dimensions, causing softmax gradients to vanish."
                },
                {
                    "slide_number": 5,
                    "title": "Empirical Results",
                    "subtitle": "Benchmark Translation Performance",
                    "bullet_points": [
                        "WMT 2014 English-German: 28.4 BLEU (+2.0 over existing ensembles)",
                        "WMT 2014 English-French: 41.8 BLEU (new single-model SOTA)",
                        "Trained in 3.5 days on 8 P100 GPUs (1/4 the compute cost of prior SOTA)",
                        "Base model reached 27.3 BLEU after only 12 hours of training"
                    ],
                    "key_takeaway": "The Transformer establishes superior translation accuracy with orders of magnitude less training time.",
                    "speaker_notes": "These empirical figures demonstrate both superior performance and dramatic computational savings over previous architectures."
                },
                {
                    "slide_number": 6,
                    "title": "Limitations & Viva Defense Preparation",
                    "subtitle": "Critical Trade-Offs & Open Questions",
                    "bullet_points": [
                        "Quadratic Complexity: O(n^2) time and memory scaling with sequence length n",
                        "Fixed Context Windows: Struggles to extrapolate to lengths significantly exceeding training",
                        "Lack of Inductive Bias: Demands massive datasets to learn positional relationships",
                        "High Peak Memory during inference due to caching key-value matrices"
                    ],
                    "key_takeaway": "Quadratic attention complexity is the primary theoretical bottleneck for long documents and high-resolution inputs.",
                    "speaker_notes": "In defense examination, committees often press on the quadratic O of n squared complexity. Candidates must be prepared to articulate sparse attention or linear attention alternatives."
                }
            ], indent=2)

        # 3. Viva Defense Questions
        if "viva" in prompt_lower or "examiner" in prompt_lower:
            return r"""### Rigorous Viva Voce Examination Questions & Model Answers

---

#### [Tier 1: Fundamental Concept Questions]
**Question 1:** *Why did the authors replace recurrent units entirely with attention mechanisms, rather than simply improving attention within an LSTM framework?*
* **Examiner Intent:** Tests if the candidate understands the fundamental hardware constraint of sequential dependency $h_t = f(h_{t-1}, x_t)$ versus global parallel matrix operations.
* **Model Answer:** Recurrence inherently forces step-by-step sequential processing along the sequence length dimension, making parallelization across tokens within a sentence mathematically impossible during training. By replacing recurrence entirely with self-attention, the Transformer computes pairwise affinities across all tokens simultaneously through matrix multiplications ($QK^T$), allowing complete parallelization across GPUs and eliminating the $O(n)$ path length bottleneck.
* **Common Trap to Avoid:** Stating vaguely that "attention is better" without explaining the parallelization bottleneck or the $O(1)$ maximum path length between distant tokens.
* **Paper Grounding:** Section 1 & Table 1 (Maximum Path Lengths & Sequential Operations).

---

#### [Tier 2: Methodological Deep-Dive]
**Question 2:** *What is the exact mathematical justification for the scaling factor $\frac{1}{\sqrt{d_k}}$ in Scaled Dot-Product Attention? What happens if you omit it?*
* **Examiner Intent:** Tests mathematical rigor regarding variance of dot products and gradient dynamics of the softmax function.
* **Model Answer:** Assume components of $q$ and $k$ are independent random variables with mean 0 and variance 1. Their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$. For large dimensions (e.g., $d_k = 64$), the dot products grow large in magnitude, driving the softmax function into regions with extremely flat gradients (saturation). Scaling by $\frac{1}{\sqrt{d_k}}$ restores the variance to 1, preventing vanishing gradients during backpropagation.
* **Common Trap to Avoid:** Confusing dot-product scaling with standard feature normalization or batch normalization.
* **Paper Grounding:** Section 3.2 (Scaled Dot-Product Attention).

---

#### [Tier 3: Skeptical & Critical Edge-Case Questions]
**Question 3:** *How do you defend the model's quadratic $O(n^2)$ computational complexity when handling ultra-long sequences such as full books or genomics sequences?*
* **Examiner Intent:** Pushes the candidate to acknowledge the real-world limitation of self-attention memory footprint and evaluate follow-up architectures.
* **Model Answer:** The $O(n^2)$ complexity arises because every token attends to every other token, requiring an $n \times n$ attention matrix. While this is optimal for moderate sequence lengths (e.g., translation sentences of $n \le 512$), it becomes intractable for documents of $n > 10,000$. In defense, one must acknowledge that while the original Transformer accepted this trade-off for translation, modern extensions adopt sparse attention (Longformer), kernelized linear attention (Performers), or state-space models (Mamba) to achieve $O(n)$ scaling.
* **Common Trap to Avoid:** Denying that $O(n^2)$ is a problem, or confusing computational time complexity with parameter count.
* **Paper Grounding:** Section 4 (Complexity per Layer analysis).

---

#### [Tier 4: Limitations, Assumptions & Future Directions]
**Question 4:** *Why are positional encodings required, and why did the authors choose sinusoidal functions over learned absolute embeddings?*
* **Examiner Intent:** Checks understanding of permutation invariance in self-attention and extrapolation capability.
* **Model Answer:** Self-attention is permutation-invariant: shuffling the order of input tokens yields the exact same attention outputs without positional markers. Sinusoidal functions were chosen because $PE_{pos+k}$ can be represented as a linear transformation of $PE_{pos}$, allowing the model to attend by relative positions and hypothesize sequence lengths longer than those observed during training. Learned positional embeddings yielded identical translation scores but lacked mathematical extrapolation properties.
* **Common Trap to Avoid:** Claiming that Transformer layers automatically understand word order like RNNs do.
* **Paper Grounding:** Section 3.5 (Positional Encoding).
"""

        # 4. Structured Summary
        return r"""<thinking>
1. Synthesizing research paper based on retrieved context.
2. Context provides complete information on Transformer architecture, multi-head attention, experimental BLEU benchmarks, and quadratic limitations.
3. Structuring into academic sections with strict grounding.
</thinking>

### 1. Executive Abstract & Core Thesis
The paper introduces the **Transformer**, a revolutionary neural sequence transduction model that dispenses with recurrence and convolutions entirely, relying solely on **stacked self-attention mechanisms**. By replacing sequential token processing with parallel matrix operations, the Transformer achieves superior translation quality while drastically cutting training duration.

### 2. Problem Formulation & Motivation
Prior state-of-the-art architectures (LSTMs, GRUs) rely on recurrent computation along symbol positions ($h_t = f(h_{t-1}, x_t)$). This sequential design introduces two fundamental bottlenecks:
1. **No parallelization across sequence length during training**, severely constraining throughput on modern parallel accelerators.
2. **Path length of $O(n)$** between distant tokens, making long-range contextual relationships difficult to learn.

### 3. Proposed Methodology & Architecture
The Transformer follows an **Encoder-Decoder** topology with $N = 6$ identical layers:
* **Multi-Head Self-Attention:** Linearly projects queries, keys, and values $h = 8$ times into subspaces ($d_k = d_v = 64$) with dimension $d_{\text{model}} = 512$.
* **Scaled Dot-Product Attention:** Formulated as $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
* **Positional Encoding:** Injects sinusoidal functions ($\sin$, $\cos$) to preserve sequence order without recurrence.
* **Feed-Forward Networks & Residuals:** Each sub-layer incorporates $\text{LayerNorm}(x + \text{Sublayer}(x))$.

### 4. Empirical Results & Benchmarks
* **WMT 2014 English-to-German:** Reached **28.4 BLEU**, establishing a new state of the art and outperforming existing ensembles by over 2.0 BLEU.
* **WMT 2014 English-to-French:** Achieved **41.8 BLEU** after 3.5 days of training on 8 P100 GPUs, a fraction of previous compute costs.

### 5. Critical Limitations & Boundary Conditions
1. **Quadratic Time and Memory Complexity ($O(n^2)$):** Limits application to very long sequences.
2. **Lack of Inductive Bias:** Requires substantial data volume compared to convolutional or recurrent networks.
3. **Hardware Barrier:** Demands multi-GPU parallel infrastructure for convergence.

### 6. Academic Significance
The Transformer established the foundational backbone for modern Large Language Models (BERT, GPT, Gemini, LLaMA), redefining natural language processing, computer vision, and computational biology.
"""

    @property
    def _llm_type(self) -> str:
        return "mock_academic_model"


class LLMFactory:
    """
    Factory creating configured LangChain LLM instances for LCEL pipelines.
    """

    @staticmethod
    def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "AgenticAI"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def get_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
        """Queries local Ollama for installed models."""
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "AgenticAI"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return models if models else config.DEFAULT_MODELS[config.PROVIDER_OLLAMA]
        except Exception:
            return config.DEFAULT_MODELS[config.PROVIDER_OLLAMA]

    @staticmethod
    def create_llm(
        provider: str,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Any:
        """
        Creates a LangChain chat model instance compatible with LCEL.
        Falls back smoothly to MockAcademicChatModel if provider is unconfigured.
        """
        # 1. Ollama (Local LLM)
        if provider == config.PROVIDER_OLLAMA:
            try:
                from langchain_ollama import ChatOllama
                m_name = model_name or config.DEFAULT_MODELS[config.PROVIDER_OLLAMA][0]
                url = base_url or "http://localhost:11434"
                if LLMFactory.is_ollama_running(url):
                    return ChatOllama(
                        model=m_name,
                        base_url=url,
                        temperature=temperature,
                        num_predict=1024
                    )
            except Exception as e:
                print(f"Ollama load warning: {e}")

        # 2. Google Gemini (API)
        elif provider == config.PROVIDER_GEMINI:
            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    m_name = model_name or "gemini-1.5-flash"
                    return ChatGoogleGenerativeAI(
                        model=m_name,
                        google_api_key=key,
                        temperature=temperature
                    )
                except Exception as e:
                    print(f"Gemini load warning: {e}")

        # 3. Groq (API - Fast)
        elif provider == config.PROVIDER_GROQ:
            key = api_key or os.getenv("GROQ_API_KEY")
            if key:
                try:
                    from langchain_openai import ChatOpenAI
                    m_name = model_name or "llama-3.3-70b-versatile"
                    return ChatOpenAI(
                        model=m_name,
                        api_key=key,
                        base_url="https://api.groq.com/openai/v1",
                        temperature=temperature
                    )
                except Exception as e:
                    print(f"Groq load warning: {e}")

        # 4. OpenAI (API)
        elif provider == config.PROVIDER_OPENAI:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if key:
                try:
                    from langchain_openai import ChatOpenAI
                    m_name = model_name or "gpt-4o-mini"
                    return ChatOpenAI(
                        model=m_name,
                        api_key=key,
                        temperature=temperature
                    )
                except Exception as e:
                    print(f"OpenAI load warning: {e}")

        # Fallback to Mock Chat Model for zero-crash graceful offline testing
        return MockAcademicChatModel()
