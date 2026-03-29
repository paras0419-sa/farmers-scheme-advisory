"""
Pipeline: Unified entry points for KisanSathi — text and voice queries.

Wires together: Retriever, Generator, Verifier, Translator, and Voice modules.

Usage:
    from src.pipeline import KisanSathiPipeline

    pipeline = KisanSathiPipeline()

    # Text query (Hindi)
    result = pipeline.text_query("PM-KISAN योजना के लिए कौन पात्र है?")
    print(result["answer"])

    # Voice query
    result = pipeline.voice_query("voice_note.ogg")
    print(result["answer"])
"""

import logging
from typing import Optional

from src.generator import SchemeGenerator
from src.retriever import SchemeRetriever
from src.translator import SchemeTranslator
from src.verifier import AnswerVerifier

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.25
NO_SCHEME_MSG_EN = (
    "I could not find a relevant government scheme for your question. "
    "Please try rephrasing, or ask about a specific scheme like PM-KISAN, "
    "PMFBY, Kisan Credit Card, etc."
)
NO_SCHEME_MSG_HI = (
    "आपके प्रश्न से संबंधित कोई सरकारी योजना नहीं मिली। "
    "कृपया दोबारा पूछें, या किसी विशेष योजना के बारे में पूछें "
    "जैसे PM-KISAN, PMFBY, किसान क्रेडिट कार्ड आदि।"
)


class KisanSathiPipeline:
    """Unified RAG pipeline with translation, verification, and voice support."""

    def __init__(self, model: str = "mistral", top_k: int = 5):
        self.retriever = SchemeRetriever()
        self.generator = SchemeGenerator(model=model)
        self.translator = SchemeTranslator(model=model)
        self.verifier = AnswerVerifier()
        self.top_k = top_k
        self._voice = None  # Lazy-load (heavy model)

    def _get_voice(self):
        """Lazy-load voice transcriber (IndicWhisper is ~1GB)."""
        if self._voice is None:
            from src.voice import VoiceTranscriber
            self._voice = VoiceTranscriber()
        return self._voice

    def text_query(self, query: str, lang: Optional[str] = None) -> dict:
        """Process a text query through the full RAG pipeline.

        Args:
            query: User's question (Hindi or English).
            lang: Language code ('hi' or 'en'). Auto-detected if None.

        Returns:
            Dict with: answer, sources, grounding, language, english_query
        """
        # Validate input
        cleaned = query.strip()
        if not cleaned or len(cleaned) < 2:
            return self._no_result(query, lang or "en", reason="empty_input")

        # Auto-detect language if not specified
        if lang is None:
            lang = self.translator.detect_language(cleaned)

        # Translate Hindi → English for retrieval
        if lang == "hi":
            english_query = self.translator.hindi_to_english(cleaned)
        else:
            english_query = cleaned

        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(english_query, top_k=self.top_k)

        # Check if retrieved chunks are relevant enough
        if not chunks or chunks[0]["score"] < RELEVANCE_THRESHOLD:
            logger.info("No relevant chunks (top score: %s)", chunks[0]["score"] if chunks else "N/A")
            return self._no_result(english_query, lang, reason="no_relevant_scheme")

        # Generate answer in English
        english_answer = self.generator.generate(english_query, chunks)

        # Verify grounding
        grounding = self.verifier.verify(english_answer, chunks)

        # Translate answer to Hindi if needed
        if lang == "hi":
            answer = self.translator.english_to_hindi(english_answer)
        else:
            answer = english_answer

        # Build source list
        sources = [
            {"scheme": c["scheme_name"], "score": c["score"]}
            for c in chunks
        ]

        return {
            "answer": answer,
            "english_answer": english_answer,
            "english_query": english_query,
            "sources": sources,
            "grounding": grounding,
            "language": lang,
        }

    def _no_result(self, query: str, lang: str, reason: str) -> dict:
        """Return a structured response when no answer can be produced."""
        answer = NO_SCHEME_MSG_HI if lang == "hi" else NO_SCHEME_MSG_EN
        return {
            "answer": answer,
            "english_answer": NO_SCHEME_MSG_EN,
            "english_query": query,
            "sources": [],
            "grounding": {"score": 0.0, "is_grounded": True, "warning": None, "answer_tokens": 0, "overlap_tokens": 0},
            "language": lang,
            "no_result_reason": reason,
        }

    def voice_query(self, audio_path: str) -> dict:
        """Process a voice query: transcribe → text pipeline.

        Args:
            audio_path: Path to audio file (.ogg or .wav).

        Returns:
            Dict with all text_query fields plus: transcription
        """
        voice = self._get_voice()
        transcription = voice.transcribe(audio_path)

        result = self.text_query(transcription, lang="hi")
        result["transcription"] = transcription
        return result


if __name__ == "__main__":
    import sys

    pipeline = KisanSathiPipeline()

    if len(sys.argv) > 1 and sys.argv[1].endswith((".ogg", ".wav", ".mp3")):
        # Voice mode
        print(f"Processing voice file: {sys.argv[1]}")
        result = pipeline.voice_query(sys.argv[1])
        print(f"\nTranscription: {result['transcription']}")
    else:
        # Text mode
        query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "PM-KISAN योजना के लिए कौन पात्र है?"
        print(f"Query: {query}")
        result = pipeline.text_query(query)

    print(f"\nLanguage: {result['language']}")
    print(f"English query: {result['english_query']}")
    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(result["answer"])
    print(f"\n{'='*60}")
    print("ENGLISH ANSWER:")
    print(f"{'='*60}")
    print(result["english_answer"])
    print(f"\nGrounding: {result['grounding']['score']:.2f} "
          f"({'Grounded' if result['grounding']['is_grounded'] else 'Low grounding'})")
    if result["grounding"]["warning"]:
        print(f"Warning: {result['grounding']['warning']}")
    print(f"\nSources:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] {s['scheme']} (score: {s['score']:.3f})")
