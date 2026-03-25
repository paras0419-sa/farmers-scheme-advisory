"""
Retriever: Embed user query and search ChromaDB for relevant scheme chunks.

Usage:
    from src.retriever import SchemeRetriever

    retriever = SchemeRetriever()
    results = retriever.retrieve("What is PM-KISAN?")
    for r in results:
        print(f"{r['score']:.3f} | {r['source']} | {r['text'][:100]}")
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "farmer_schemes"


class SchemeRetriever:
    """Retrieves relevant scheme chunks from ChromaDB using semantic search."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.model = SentenceTransformer(embedding_model)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Embed query and return top-K relevant chunks with scores.

        Returns:
            List of dicts with keys: text, source, scheme_name, score, chunk_index
        """
        query_embedding = self.model.encode([query])[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        chunks = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "source": metadata["source_file"],
                    "scheme_name": metadata["scheme_name"],
                    "score": round(1 / (1 + distance), 4),
                    "chunk_index": metadata["chunk_index"],
                }
            )

        return chunks


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is PM-KISAN?"
    retriever = SchemeRetriever()
    results = retriever.retrieve(query)

    print(f"\nQuery: '{query}'")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.3f} | Scheme: {r['scheme_name']}")
        print(f"    Source: {r['source']}")
        print(f"    {r['text'][:200]}...")
        print()
