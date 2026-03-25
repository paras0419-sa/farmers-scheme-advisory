# KisanSathi — Farmer Scheme Advisory

An AI-powered RAG (Retrieval-Augmented Generation) system that ingests Indian government scheme documents for farmers and answers queries using local LLMs. Built to eventually serve farmers via Telegram in vernacular languages.

## Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Retriever   │────▶│  ChromaDB     │────▶│  Generator   │
│  (MiniLM     │     │  (Vector      │     │  (Mistral 7B │
│   Embeddings)│     │   Store)      │     │   via Ollama)│
└──────────────┘     └───────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                           Grounded Answer
                                           + Source Citations
```

**How it works:**

1. **Ingestion** — Government scheme PDFs/docs are chunked by section, prefixed with scheme context, embedded using multilingual MiniLM, and stored in ChromaDB
2. **Retrieval** — User query is embedded with the same model, and top-5 most similar chunks are retrieved via cosine similarity
3. **Generation** — Retrieved chunks are injected into a grounded prompt, and Mistral 7B (via Ollama) generates an answer constrained to only use the provided context

## Project Structure

```
farmers-scheme-advisory/
├── scripts/
│   ├── ingest.py           # Document ingestion pipeline
│   └── query.py            # End-to-end RAG query CLI
├── src/
│   ├── __init__.py
│   ├── retriever.py        # Semantic search against ChromaDB
│   └── generator.py        # Grounded generation via Ollama/Mistral
├── data/
│   └── farmers-scheme/     # 16 government scheme documents (.md)
├── chroma_db/              # ChromaDB persistent vector store
├── Day3Learning.md         # Learning guide for RAG concepts
├── future-extensions/      # PRD, execution plan, full roadmap
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed locally

### Installation

```bash
# Clone the repo
git clone https://github.com/paras0419-sa/farmers-scheme-advisory.git
cd farmers-scheme-advisory

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Mistral 7B model (~4.4 GB)
ollama pull mistral
```

### Running

```bash
# Step 1: Ingest scheme documents into ChromaDB
python scripts/ingest.py

# Step 2: Query the RAG pipeline
python scripts/query.py "What is PM-KISAN and who is eligible?"

# Interactive mode
python scripts/query.py --interactive

# Debug retrieval quality
python scripts/query.py "query here" --show-chunks

# Adjust number of retrieved chunks
python scripts/query.py "query here" --top-k 3
```

## Scheme Coverage

16 Indian government schemes ingested:

| #   | Scheme                           | Category                          |
| --- | -------------------------------- | --------------------------------- |
| 1   | PM-KISAN Samman Nidhi            | Direct income support             |
| 2   | PMFBY (Fasal Bima Yojana)        | Crop insurance                    |
| 3   | Kisan Credit Card (KCC)          | Agricultural credit               |
| 4   | PM-KUSUM                         | Solar energy for farmers          |
| 5   | Soil Health Card                 | Soil testing & advisory           |
| 6   | eNAM                             | Online agricultural market        |
| 7   | Agriculture Infrastructure Fund  | Infrastructure loans              |
| 8   | PM Dhan-Dhaanya Krishi Yojana    | Agricultural development          |
| 9   | Paramparagat Krishi Vikas Yojana | Organic farming                   |
| 10  | PM Krishi Sinchayee Yojana       | Irrigation (Per Drop More Crop)   |
| 11  | RKVY                             | State agricultural development    |
| 12  | National Food Security Mission   | Food grain production             |
| 13  | SMAM                             | Agricultural mechanization        |
| 14  | NMEO Oil Palm                    | Oil palm cultivation              |
| 15  | MISS                             | Interest subvention on crop loans |
| 16  | Mission Aatmanirbharta           | Pulses self-sufficiency           |

## Design Decisions & Fixes

### Cosine distance instead of L2 (default)

ChromaDB defaults to L2 (Euclidean) distance for similarity search. We switched to cosine distance because:

- Text embeddings from sentence-transformers are normalized for cosine similarity, not L2
- L2 produced scores in the range of 16-18 (hard to interpret), while cosine gives clean 0-1 similarity scores
- Cosine similarity better captures semantic meaning for text retrieval tasks

### Scheme-name prefix on every chunk

Each chunk is prepended with the scheme name and section heading (e.g., `PM Kisan Samman Nidhi (PM-KISAN) — Eligibility Criteria:`). This was added because:

- **Problem:** The query "What is the eligibility for PM-KISAN?" was returning the "How to Apply" section instead of the "Eligibility Criteria" section
- **Root cause:** The Eligibility chunk was too short (3 bullet points) with no keyword overlap beyond the heading. The embedding model scored "How to Apply" higher because it contained richer text about registration steps
- **Fix:** Prepending scheme name + section label gives the embedding model more semantic signal about what each chunk is about
- **Result:** Eligibility chunks now rank correctly for eligibility queries, and scheme-specific queries are much more precise

### Section-based chunking over fixed-size chunking

Documents are split by `##` markdown headings rather than arbitrary character boundaries. This preserves semantic coherence — each chunk represents one complete topic (e.g., Benefits, Eligibility, How to Apply) rather than cutting mid-sentence. Large sections fall back to overlapping character-based chunking.

## Tech Stack

| Component    | Technology                              | Why                                                             |
| ------------ | --------------------------------------- | --------------------------------------------------------------- |
| Embeddings   | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual support for future Hindi/regional language queries |
| Vector Store | ChromaDB (persistent, cosine distance)  | Simple, embedded, no server needed for MVP                      |
| LLM          | Mistral 7B via Ollama                   | Free, local, no API key needed, good quality for grounded Q&A   |
| Language     | Python 3.9                              | Ecosystem support for ML/NLP libraries                          |

## What's Next

- **Day 4:** Verification pipeline (answer grounding check) + Hindi translation layer
- **Day 5:** Telegram bot integration
- **Day 6:** Testing & prompt tuning
- **Day 7:** Deploy & publish

## License

See [LICENSE](LICENSE) file.
