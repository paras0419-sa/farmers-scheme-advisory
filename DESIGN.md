# Farmers Scheme Advisory System - Design Document

## Overview

An AI-powered system that ingests government scheme documents for farmers and answers
queries in vernacular languages via voice/text on platforms like Telegram.

## Requirements

### Functional

- Takes voice/text query from farmers in vernacular language
- Tells the farmer which schemes are useful for them
- Provides text summary of schemes with benefits
- Connects to Telegram or other messaging platforms

### Non-Functional

- Low latency / fast inference
- Small language model
- Verify before providing each query output

---

## System Components

```text
┌─────────────────────────────────────────────────────┐
│                   FARMER (User)                     │
│              Voice / Text in Hindi etc.             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  A. MESSAGING INTERFACE (Telegram Bot)               │
│     - Receives text/voice messages                   │
│     - Sends back scheme summaries                    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  B. SPEECH-TO-TEXT (ASR) — only for voice input      │
│     - Converts vernacular speech → text              │
│     - Options: Whisper, Google Speech API,            │
│       IndicWhisper (best for Indian languages)       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  C. QUERY UNDERSTANDING / TRANSLATION                │
│     - Translate vernacular text → English (or keep   │
│       multilingual if model supports it)             │
│     - Extract farmer's context: crop, location,      │
│       land size, income bracket                      │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  D. RETRIEVAL ENGINE (RAG Pipeline)                  │
│     - Vector DB (FAISS / Chroma / Qdrant) stores     │
│       chunked scheme documents as embeddings         │
│     - Retrieves top-K relevant scheme chunks         │
│       based on query embedding similarity            │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  E. LLM / SLM (Answer Generation)                   │
│     - Small Language Model: Phi-3-mini, Gemma 2B,    │
│       or Mistral 7B (quantized)                      │
│     - Takes retrieved chunks + query → generates     │
│       a grounded, verified answer                    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  F. VERIFICATION LAYER                               │
│     - Checks that the generated answer is grounded   │
│       in the retrieved documents (no hallucination)  │
│     - Simple approach: citation matching / overlap    │
│       scoring between answer and source chunks       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  G. RESPONSE FORMATTER + TRANSLATION                 │
│     - Translate English answer → farmer's language   │
│     - Format as a clean Telegram message             │
│     - Optional: Text-to-Speech for voice reply       │
└──────────────────────────────────────────────────────┘
```

### Offline / Background Components

| Component                 | Purpose                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **H. Document Ingestion** | Scrape/collect govt scheme PDFs/web pages, parse, chunk, embed, store in vector DB. Run when schemes update.  |
| **I. Embedding Model**    | Multilingual sentence transformer (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) used at ingestion and query. |

---

## Topics to Study

| Area                           | What to Learn                                           | Why                                      |
| ------------------------------ | ------------------------------------------------------- | ---------------------------------------- |
| **RAG**                        | Chunking, embeddings, vector DB, retrieval              | Core architecture pattern                |
| **Vector Databases**           | FAISS, ChromaDB, Qdrant                                 | Retrieval speed and accuracy             |
| **Small Language Models**      | Phi-3, Gemma, Mistral, quantization (GGUF/GPTQ), Ollama | Small model + low latency requirement    |
| **Prompt Engineering**         | System prompts, few-shot, constraining to context       | Controls quality, prevents hallucination |
| **Speech-to-Text (ASR)**       | Whisper, IndicWhisper, Telegram audio handling          | Voice input requirement                  |
| **Multilingual NLP**           | IndicTrans, Google Translate, multilingual embeddings   | Vernacular language support              |
| **Telegram Bot API**           | python-telegram-bot library, message/voice handling     | User-facing interface                    |
| **Document Parsing**           | PyMuPDF, pdfplumber (or Apache Tika for Java)           | Govt schemes come as PDFs/web pages      |
| **LLM Evaluation & Grounding** | Answer faithfulness, RAGAS framework, citation overlap  | Verify before providing output           |

---

## Tech Stack (MVP)

| Layer            | Choice                                        | Rationale                         |
| ---------------- | --------------------------------------------- | --------------------------------- |
| Language         | **Python** (MVP), Java later                  | Best ML/AI ecosystem              |
| Bot Framework    | `python-telegram-bot`                         | Mature, well-documented           |
| ASR (v2)         | OpenAI Whisper / IndicWhisper                 | Best open-source multilingual ASR |
| Embeddings       | `paraphrase-multilingual-MiniLM-L12-v2`       | Free, multilingual, fast          |
| Vector Store     | ChromaDB                                      | Simple, runs locally, no infra    |
| LLM              | Ollama + Mistral 7B (quantized) OR Claude API | Local/free or higher quality      |
| Document Parsing | PyMuPDF / pdfplumber                          | Extract text from govt PDFs       |
| Translation      | IndicTrans2 or Google Translate API           | Hindi ↔ English                   |

---

## MVP Scope

| In Scope                           | Out of Scope (v2+)        |
| ---------------------------------- | ------------------------- |
| Text queries in Hindi              | Voice input               |
| 10-20 scheme documents manually    | Automated scraping        |
| Single retrieval + generation pass | Multi-turn conversation   |
| Telegram only                      | WhatsApp, web UI          |
| One language (Hindi)               | Full multilingual support |

---

## Project Structure

```text
farmers-app/
├── data/
│   └── farmers-schemes/              # Raw PDFs and parsed text
├── scripts/
│   └── ingest.py             # Parse docs → chunks → ChromaDB
├── src/
│   ├── bot.py                # Telegram bot entry point
│   ├── retriever.py          # Embedding + ChromaDB search
│   ├── generator.py          # LLM prompt + call
│   ├── verifier.py           # Answer grounding check
│   └── translator.py         # Hindi ↔ English
├── requirements.txt
├── .env                      # API keys (not committed)
├── .gitignore
└── README.md
```

---

## ML Model Strategy

For MVP, **no training from scratch**:

| Step              | Approach                                                                                       |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| Embedding model   | Pre-trained multilingual sentence transformer. Just load and call `model.encode(text)`.        |
| LLM               | Pre-trained Mistral 7B quantized via Ollama. Prompt engineering + RAG for grounded answers.    |
| Fine-tuning (v2+) | Collect QA pairs from real interactions. Fine-tune with LoRA/QLoRA for domain accuracy.        |
| Deployment        | Ollama on a small cloud VM (4GB+ VRAM). ChromaDB in-process. Telegram bot as long-polling app. |

**Key insight:** RAG lets you get accurate, document-grounded answers without training. The model
generates based on retrieved context, not memorized knowledge.
