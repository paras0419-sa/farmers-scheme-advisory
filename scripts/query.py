"""
End-to-end RAG query script: Query → Retrieve → Generate → Answer.

Usage:
    python scripts/query.py "What is PM-KISAN and who is eligible?"
    python scripts/query.py --interactive
    python scripts/query.py "query" --top-k 3
    python scripts/query.py "query" --show-chunks
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import SchemeRetriever
from src.generator import SchemeGenerator


def run_query(
    query: str,
    retriever: SchemeRetriever,
    generator: SchemeGenerator,
    top_k: int = 5,
    show_chunks: bool = False,
) -> str:
    """Run a single query through the RAG pipeline."""
    chunks = retriever.retrieve(query, top_k=top_k)

    if show_chunks:
        print("\n--- Retrieved Chunks ---")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] Score: {c['score']:.3f} | {c['scheme_name']}")
            print(f"    {c['text'][:300]}")
        print("\n--- Generating Answer ---\n")

    answer = generator.generate(query, chunks)

    print(answer)
    print(f"\n📄 Sources:")
    for i, c in enumerate(chunks[:3], 1):
        print(f"   [{i}] {c['scheme_name']} (relevance: {c['score']:.3f})")

    return answer


def interactive_mode(retriever: SchemeRetriever, generator: SchemeGenerator, top_k: int, show_chunks: bool):
    """REPL mode for testing multiple queries."""
    print("=" * 60)
    print("KisanSathi RAG Pipeline — Interactive Mode")
    print("Type your question and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        run_query(query, retriever, generator, top_k=top_k, show_chunks=show_chunks)


def main():
    parser = argparse.ArgumentParser(description="Query the KisanSathi RAG pipeline")
    parser.add_argument("query", nargs="?", help="The question to ask")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive REPL mode")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--show-chunks", action="store_true", help="Show retrieved chunks before the answer")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model name (default: mistral)")
    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.error("Provide a query or use --interactive mode")

    print("Loading models...")
    retriever = SchemeRetriever()
    generator = SchemeGenerator(model=args.model)
    print("Ready.\n")

    if args.interactive:
        interactive_mode(retriever, generator, args.top_k, args.show_chunks)
    else:
        run_query(args.query, retriever, generator, top_k=args.top_k, show_chunks=args.show_chunks)


if __name__ == "__main__":
    main()
