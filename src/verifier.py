"""
Verifier: Check if the generated answer is grounded in the retrieved source chunks.

Uses token overlap scoring — compares answer tokens against source chunk tokens
to detect potential hallucination.

Usage:
    from src.verifier import AnswerVerifier

    verifier = AnswerVerifier()
    result = verifier.verify("PM-KISAN provides ₹6000 annually", chunks)
    print(f"Grounded: {result['is_grounded']} (score: {result['score']:.2f})")
"""

import re

# Common English stopwords to exclude from overlap calculation
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom",
}

GROUNDING_THRESHOLD = 0.4


class AnswerVerifier:
    """Verifies that generated answers are grounded in source chunks."""

    def __init__(self, threshold: float = GROUNDING_THRESHOLD):
        self.threshold = threshold

    def _tokenize(self, text: str) -> set[str]:
        """Lowercase, extract words, remove stopwords."""
        words = re.findall(r"[a-zA-Z0-9₹\u0900-\u097F]+", text.lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 1}

    def verify(self, answer: str, chunks: list[dict]) -> dict:
        """Check answer grounding against source chunks.

        Args:
            answer: The generated answer text.
            chunks: List of retrieved chunk dicts (must have 'text' key).

        Returns:
            Dict with keys: score, is_grounded, warning, answer_tokens, overlap_tokens
        """
        answer_tokens = self._tokenize(answer)

        # Combine all chunk texts into one token set
        chunk_text = " ".join(c["text"] for c in chunks)
        chunk_tokens = self._tokenize(chunk_text)

        if not answer_tokens:
            return {
                "score": 0.0,
                "is_grounded": False,
                "warning": "Answer has no meaningful tokens to verify.",
                "answer_tokens": 0,
                "overlap_tokens": 0,
            }

        overlap = answer_tokens & chunk_tokens
        score = len(overlap) / len(answer_tokens)

        is_grounded = score >= self.threshold
        warning = None if is_grounded else (
            f"Low grounding score ({score:.2f}). "
            "The answer may contain information not found in the source documents."
        )

        return {
            "score": round(score, 4),
            "is_grounded": is_grounded,
            "warning": warning,
            "answer_tokens": len(answer_tokens),
            "overlap_tokens": len(overlap),
        }


if __name__ == "__main__":
    # Quick test with fake data
    test_chunks = [
        {"text": "PM-KISAN provides ₹6000 annually to small and marginal farmers with land holdings."},
        {"text": "Eligibility: All farmer families with cultivable land are eligible for PM-KISAN."},
    ]

    verifier = AnswerVerifier()

    # Test 1: Grounded answer
    result = verifier.verify(
        "PM-KISAN provides ₹6000 annually to farmers with land.",
        test_chunks,
    )
    print(f"[Grounded test]  Score: {result['score']:.2f} | Grounded: {result['is_grounded']}")

    # Test 2: Hallucinated answer
    result = verifier.verify(
        "The government gives ₹50000 for tractor purchase under PM Tractor scheme.",
        test_chunks,
    )
    print(f"[Hallucinated]   Score: {result['score']:.2f} | Grounded: {result['is_grounded']}")
