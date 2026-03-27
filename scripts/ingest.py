"""
Ingestion script: Parse scheme documents, chunk text, embed, and store in ChromaDB.

Usage:
    python scripts/ingest.py                    # Ingest all docs from data/farmers-scheme/
    python scripts/ingest.py --query "kisan"    # Ingest + run a test query
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "farmers-scheme"
CHROMA_DIR = ROOT_DIR / "chroma_db"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "farmer_schemes"
CHUNK_SIZE = 700  # characters (not tokens — simpler and close enough for MVP)
CHUNK_OVERLAP = 200


def load_documents(data_dir: Path) -> list[dict]:
    """Load all .md files from the data directory."""
    docs = []
    for filepath in sorted(data_dir.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        docs.append({
            "filename": filepath.name,
            "text": text,
            "scheme_name": extract_scheme_name(text, filepath.name),
        })
    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs


def extract_scheme_name(text: str, filename: str) -> str:
    """Extract scheme name from the first heading or fallback to filename."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return filename.replace(".md", "").replace("-", " ").title()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def chunk_by_sections(text: str, scheme_name: str) -> list[str]:
    """Split markdown into sections based on ## headings. Falls back to character chunking for large sections.
    Prepends scheme name context to each chunk for better embedding quality."""
    sections = []
    current_section = []
    current_heading = ""

    for line in text.splitlines():
        if line.startswith("## "):
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    sections.append(section_text)
            current_heading = line
            current_section = [current_heading]
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        section_text = "\n".join(current_section).strip()
        if section_text:
            sections.append(section_text)

    # Prepend scheme name context and sub-chunk large sections
    final_chunks = []
    for section in sections:
        # Extract section heading if present
        first_line = section.split("\n", 1)[0].strip()
        if first_line.startswith("## "):
            section_label = first_line[3:].strip()
            prefix = f"{scheme_name} — {section_label}:\n"
        elif first_line.startswith("# "):
            prefix = ""  # Header chunk already contains the scheme name
        else:
            prefix = f"{scheme_name}:\n"

        prefixed = prefix + section if prefix else section

        if len(prefixed) > CHUNK_SIZE * 2:
            final_chunks.extend(chunk_text(prefixed))
        else:
            final_chunks.append(prefixed)

    return final_chunks


def build_chunks(docs: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """Build chunk IDs, texts, and metadata from documents."""
    all_ids = []
    all_texts = []
    all_metadatas = []

    for doc in docs:
        chunks = chunk_by_sections(doc["text"], doc["scheme_name"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['filename']}::chunk_{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append({
                "source_file": doc["filename"],
                "scheme_name": doc["scheme_name"],
                "chunk_index": i,
            })

    print(f"Created {len(all_texts)} chunks from {len(docs)} documents")
    return all_ids, all_texts, all_metadatas


def ingest(data_dir: Path = DATA_DIR, chroma_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    """Main ingestion pipeline: load docs → chunk → embed → store in ChromaDB."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    docs = load_documents(data_dir)
    if not docs:
        print("No documents found. Exiting.")
        sys.exit(1)

    chunk_ids, chunk_texts, chunk_metadatas = build_chunks(docs)

    print(f"Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True)

    print(f"Storing in ChromaDB at {chroma_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if it exists, to allow re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Indian government farmer scheme documents",
            "hnsw:space": "cosine",
        },
    )

    # ChromaDB has a batch limit, add in batches of 100
    batch_size = 100
    for i in range(0, len(chunk_ids), batch_size):
        end = min(i + batch_size, len(chunk_ids))
        collection.add(
            ids=chunk_ids[i:end],
            documents=chunk_texts[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=chunk_metadatas[i:end],
        )

    print(f"Ingestion complete. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")
    return collection


def test_query(query: str, chroma_dir: Path = CHROMA_DIR):
    """Run a test query against the ingested data."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    print(f"\n--- Query: '{query}' ---")
    print(f"Top {len(results['documents'][0])} results:\n")
    for i, (doc, metadata, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"[{i+1}] Similarity: {1 - distance:.3f} | Source: {metadata['scheme_name']}")
        print(f"    {doc[:200]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest farmer scheme documents into ChromaDB")
    parser.add_argument("--query", type=str, help="Run a test query after ingestion")
    parser.add_argument("--query-only", type=str, help="Only run a query (skip ingestion)")
    args = parser.parse_args()

    if args.query_only:
        test_query(args.query_only)
    else:
        ingest()
        if args.query:
            test_query(args.query)
