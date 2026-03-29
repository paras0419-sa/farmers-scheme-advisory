"""
Generator: Build a grounded prompt from retrieved chunks and generate an answer via Ollama/Mistral.

Usage:
    from src.generator import SchemeGenerator

    generator = SchemeGenerator()
    answer = generator.generate("What is PM-KISAN?", retrieved_chunks)
    print(answer)
"""

import ollama
from ollama import ResponseError

SYSTEM_PROMPT = """You are KisanSathi, an agricultural scheme advisor for Indian farmers.
Your job is to help farmers understand government schemes they may be eligible for.

RULES:
1. Answer the farmer's question using ONLY the provided context below.
2. If the context does not contain enough information to answer, say: "I don't have information about this in my database."
3. Always cite the scheme name when referencing a specific scheme.
4. Keep your answer clear, practical, and under 300 words.
5. If multiple schemes are relevant, briefly mention each one.
6. Do NOT make up facts, eligibility criteria, or amounts that are not in the context."""


class SchemeGenerator:
    """Generates grounded answers using Mistral via Ollama."""

    def __init__(self, model: str = "mistral"):
        self.model = model

    def _build_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context block for the prompt."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['scheme_name']}]\n{chunk['text']}"
            )
        return "\n\n".join(context_parts)

    def generate(self, query: str, chunks: list[dict], temperature: float = 0.2) -> str:
        """Generate a grounded answer from retrieved chunks.

        Args:
            query: The farmer's question.
            chunks: List of retrieved chunk dicts (from SchemeRetriever).
            temperature: LLM temperature (lower = more factual).

        Returns:
            The generated answer string.
        """
        context = self._build_context(chunks)

        user_message = f"""Context:
{context}

---
Farmer's Question: {query}

Answer the question using only the context above. Cite scheme names."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": temperature, "num_predict": 512},
            )
            return response["message"]["content"]
        except ResponseError as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e
        except Exception as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise TimeoutError("LLM is not responding. Is Ollama running?") from e
            raise


if __name__ == "__main__":
    from src.retriever import SchemeRetriever

    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is PM-KISAN?"

    print(f"Query: '{query}'")
    print("Retrieving relevant chunks...")
    retriever = SchemeRetriever()
    chunks = retriever.retrieve(query)

    print(f"Found {len(chunks)} chunks. Generating answer with Mistral...\n")
    generator = SchemeGenerator()
    answer = generator.generate(query, chunks)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("\n" + "=" * 60)
    print("SOURCES:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c['scheme_name']} (score: {c['score']:.3f})")
