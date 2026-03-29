"""
Translator: Hindi ↔ English translation using Mistral via Ollama.

Translates farmer queries from Hindi to English for retrieval,
and translates English answers back to Hindi for the user.

Usage:
    from src.translator import SchemeTranslator

    translator = SchemeTranslator()
    english = translator.hindi_to_english("PM-KISAN योजना के लिए कौन पात्र है?")
    hindi = translator.english_to_hindi("PM-KISAN provides ₹6000 annually.")
"""

import ollama
from ollama import ResponseError


class SchemeTranslator:
    """Translates between Hindi and English using Mistral via Ollama."""

    def __init__(self, model: str = "mistral"):
        self.model = model

    def hindi_to_english(self, text: str) -> str:
        """Translate Hindi text to English for retrieval.

        Args:
            text: Hindi text to translate.

        Returns:
            English translation string.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Hindi to English translator. "
                            "Translate the given Hindi text to English. "
                            "Output ONLY the English translation, nothing else. "
                            "Do not add explanations, notes, or formatting. "
                            "Preserve proper nouns and scheme names as-is."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                options={"temperature": 0.1, "num_predict": 256},
            )
            return response["message"]["content"].strip()
        except (ResponseError, Exception) as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise TimeoutError("Translation failed. Is Ollama running?") from e
            raise

    def english_to_hindi(self, text: str) -> str:
        """Translate English text to Hindi for the user.

        Args:
            text: English text to translate.

        Returns:
            Hindi translation string.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an English to Hindi translator. "
                            "Translate the given English text to Hindi (Devanagari script). "
                            "Output ONLY the Hindi translation, nothing else. "
                            "Do not add explanations, notes, or formatting. "
                            "Preserve proper nouns, scheme names, and numbers as-is. "
                            "Use simple Hindi that a farmer would understand."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                options={"temperature": 0.1, "num_predict": 512},
            )
            return response["message"]["content"].strip()
        except (ResponseError, Exception) as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise TimeoutError("Translation failed. Is Ollama running?") from e
            raise

    def detect_language(self, text: str) -> str:
        """Simple heuristic to detect if text is Hindi or English.

        Returns:
            'hi' if Hindi (Devanagari script detected), 'en' otherwise.
        """
        devanagari_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        total_alpha = sum(1 for c in text if c.isalpha() or "\u0900" <= c <= "\u097F")
        if total_alpha == 0:
            return "en"
        return "hi" if devanagari_count / total_alpha > 0.3 else "en"


if __name__ == "__main__":
    import sys

    translator = SchemeTranslator()

    # Test Hindi → English
    hindi_queries = [
        "PM-KISAN योजना के लिए कौन पात्र है?",
        "मुझे फसल बीमा के बारे में बताइए",
        "किसान क्रेडिट कार्ड कैसे बनवाएं?",
    ]

    print("=== Hindi → English ===\n")
    for q in hindi_queries:
        en = translator.hindi_to_english(q)
        lang = translator.detect_language(q)
        print(f"[{lang}] {q}")
        print(f"[en] {en}\n")

    # Test English → Hindi
    print("=== English → Hindi ===\n")
    en_text = "PM-KISAN provides ₹6000 annually to all farmer families with cultivable land."
    hi = translator.english_to_hindi(en_text)
    print(f"[en] {en_text}")
    print(f"[hi] {hi}")
