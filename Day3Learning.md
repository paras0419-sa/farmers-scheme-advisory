# Day 3 Execution Plan — Retrieval + Generation Pipeline

**Date:** 2026-03-25
**LLM Backend:** Ollama + Mistral 7B (local)
**Goal:** Type an English question in terminal → get a grounded scheme answer back

---

## What You're Building Today

```
User Query → [Retriever] → Top-5 Chunks from ChromaDB → [Generator] → Grounded Answer
```

Three source files + one test script:

1. `src/__init__.py` — Empty (makes src a Python package)
2. `src/retriever.py` — Query embedding + ChromaDB search
3. `src/generator.py` — Prompt construction + Mistral call via Ollama
4. `scripts/query.py` — Terminal interface for end-to-end testing

---

## Task Breakdown (Estimated: 3-4 hours hands-on)

### Task 1: `src/retriever.py` (~30 min)

**What it does:** Takes a user query string, embeds it using the same MiniLM model
from ingestion, searches ChromaDB, returns top-5 chunks with metadata.

**Key decisions:**

- Reuse `paraphrase-multilingual-MiniLM-L12-v2` (same model used in
  ingestion — MUST match)
- Return top-5 results (tunable later on Day 6)
- Return structured output: list of dicts with `text`, `source`, `score`

**Implementation outline:**

```python
class SchemeRetriever:
    def __init__(self, chroma_dir, collection_name, embedding_model):
        # Load SentenceTransformer model
        # Connect to ChromaDB PersistentClient
        # Get collection

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # Encode query
        # Query ChromaDB
        # Return list of {text, source, score}
```

---

### Task 2: Set Up Ollama + Mistral (~30 min)

Ollama is installed (v0.18.2), Mistral is pulled (4.4 GB). Verify it works:

```bash
# Quick test from terminal
ollama run mistral "What is PM-KISAN?"

# Python integration test
python -c "import ollama; print(ollama.chat(model='mistral', messages=[{'role':'user','content':'Hello'}])['message']['content'])"
```

Install the Python client:

```bash
pip install ollama
```

---

### Task 3: `src/generator.py` (~45 min)

**What it does:** Takes retrieved chunks + user query, constructs a grounded prompt,
calls Mistral via Ollama, returns the answer.

**Key decisions:**

- System prompt: instruct LLM to answer ONLY from provided context (no
  hallucination)
- Include source citations in the response
- Use Ollama Python client for Mistral calls

**Implementation outline:**

```python
SYSTEM_PROMPT = """You are KisanSathi, an agricultural scheme advisor for Indian
farmers. Answer the farmer's question using ONLY the provided context. If the
context doesn't contain the answer, say so. Cite the scheme name in your answer."""

class SchemeGenerator:
    def __init__(self, model="mistral"):
        # Store model name for Ollama

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        # Build prompt: system + context chunks + user query
        # Call Ollama chat API
        # Return answer string
```

---

### Task 4: End-to-End Test Script (~30 min)

Create `scripts/query.py` — a terminal interface to test the full pipeline:

```bash
python scripts/query.py "What is PM-KISAN and who is eligible?"
python scripts/query.py --interactive  # REPL mode
```

**What it does:**

1. Initialize Retriever + Generator
2. Retrieve relevant chunks for the query
3. Generate grounded answer via Mistral
4. Print answer + source citations to terminal

---

### Task 5: Verify & Debug (~30 min)

Test with these queries:

| #   | Query                                             | Expected Source     |
| --- | ------------------------------------------------- | ------------------- |
| 1   | "What is PM-KISAN and who is eligible?"           | PM-KISAN scheme doc |
| 2   | "How can a farmer get crop insurance?"            | PMFBY scheme doc    |
| 3   | "What schemes are available for small farmers?"   | Multiple schemes    |
| 4   | "Tell me about soil health card"                  | SHC scheme doc      |
| 5   | "What is the interest rate on Kisan Credit Card?" | KCC scheme doc      |

**Check for:**

- Are the retrieved chunks relevant? (retriever quality)
- Is the answer grounded in the chunks? (no hallucination)
- Does it cite the scheme name? (attribution)
- Is the response coherent and helpful? (generation quality)

---

## Order of Execution

```
1. Create src/__init__.py
2. Build src/retriever.py → test it standalone
3. pip install ollama → verify Mistral responds
4. Build src/generator.py → test it standalone
5. Build scripts/query.py → test end-to-end
6. Run 5 test queries, fix issues
7. Update requirements.txt with ollama dependency
```

---

## Exit Criteria (from TODO.md)

> Can type an English question in terminal and get a grounded scheme answer.

- [ ] `src/retriever.py` returns relevant chunks for scheme queries
- [ ] Mistral 7B responds via Ollama Python client
- [ ] `src/generator.py` produces grounded answers from retrieved context
- [ ] `scripts/query.py` works end-to-end from terminal
- [ ] Tested with 5+ diverse queries

---

## File Structure After Today

```
farmers-app/
├── scripts/
│   ├── ingest.py          # (Day 1-2, done)
│   └── query.py           # NEW — terminal test script
├── src/
│   ├── __init__.py        # NEW
│   ├── retriever.py       # NEW — embed query + search ChromaDB
│   └── generator.py       # NEW — prompt + Ollama/Mistral call
├── chroma_db/             # (Day 1-2, done)
├── data/farmers-scheme/   # (Day 1-2, done)
└── requirements.txt       # UPDATE — add ollama
```

---

## What You'll Learn Hands-On Today

### 1. RAG (Retrieval-Augmented Generation) Pattern

The core architecture you're implementing. This is the most important concept for
Day 3.

**What it is:** Instead of asking an LLM to answer from its training data (which may
be wrong or outdated), you first _retrieve_ relevant documents from your own
database, then feed those documents as context to the LLM and ask it to answer
_only_ from that context.

**Why it matters:**

- Eliminates hallucination — the LLM can only use facts you provide
- Works with private/domain-specific data the LLM was never trained on
- You control the knowledge base (update scheme PDFs → answers update immediately)

**The trade-off:** `top_k` (how many chunks you retrieve):

- Too low (1-2): might miss relevant info across multiple chunks
- Too high (10+): irrelevant chunks add noise, confuse the LLM, and slow it down
- Sweet spot for MVP: 5

**Hands-on experiment:**

```bash
# After building retriever, try different top_k values:
python scripts/query.py "What is PM-KISAN?" --top-k 1
python scripts/query.py "What is PM-KISAN?" --top-k 10
# Compare answer quality — see how top_k affects grounding
```

---

### 2. Semantic Search with Embeddings (Query Side)

You embedded documents in Day 2. Today you learn the _query_ side of the same
process.

**Key concept:** The same embedding model must encode both documents and queries.
You used `paraphrase-multilingual-MiniLM-L12-v2` during ingestion — the retriever
MUST use the exact same model, or similarity scores will be meaningless.

**How ChromaDB search works:**

1. Your query text → embedding vector (384 dimensions)
2. ChromaDB computes cosine similarity between query vector and every stored chunk
   vector
3. Returns the top-K most similar chunks

**Distance vs similarity:**

- ChromaDB returns `distance` (lower = more similar)
- Similarity = `1 - distance`
- A score of 0.85+ usually means high relevance
- A score below 0.5 usually means the chunk isn't relevant

**Hands-on experiment:**

```python
# In your retriever, print raw scores:
results = retriever.retrieve("What is PM-KISAN?")
for r in results:
    print(f"Score: {r['score']:.3f} | Source: {r['source']}")

# Now try a query with NO relevant answer in your data:
results = retriever.retrieve("How to cook biryani?")
# Observe: scores will be much lower — this tells you retrieval failed
```

---

### 3. Prompt Engineering for Grounded Generation

The most critical skill for building trustworthy AI applications.

**System prompt design — the 3 rules:**

1. **Role:** "You are KisanSathi, a scheme advisor..."
2. **Grounding constraint:** "Answer ONLY from the provided context"
3. **Fallback behavior:** "If the context doesn't contain the answer, say 'I don't
   have information about this'"

**Context injection format:** How you format the retrieved chunks inside the prompt
matters. A clear format helps the LLM parse the context:

```
Context:
[Source: PM-KISAN Scheme]
<chunk text here>

[Source: PMFBY Scheme]
<chunk text here>

Question: <user query>
Answer:
```

**Hands-on experiment:**

```bash
# Test 1: WITH grounding instruction
# → Ask "What schemes exist for buying tractors?" (not in your data)
# → Should respond: "I don't have information about tractor schemes"

# Test 2: WITHOUT grounding instruction (remove the system prompt constraint)
# → Same question → LLM will likely hallucinate a fake scheme

# This contrast shows you WHY grounding instructions matter
```

---

### 4. Running Local LLMs with Ollama

Ollama lets you run LLMs locally — no API keys, no cost, full privacy.

**Key concepts:**

- **Quantization:** Mistral 7B is ~4.4 GB because it's quantized (Q4). Full
  precision would be ~14 GB. Quantization trades minor quality loss for 3x less
  memory.
- **Context window:** Mistral 7B supports 8K tokens (~6K words). Your prompt
  (system + 5 chunks + query) must fit within this.
- **Temperature:** Controls randomness. For factual Q&A, use low temperature (0.1-0.3).
  For creative tasks, use higher (0.7-1.0).

**Ollama Python client:**

```python
import ollama

response = ollama.chat(
    model='mistral',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is PM-KISAN?'}
    ],
    options={'temperature': 0.2}
)
print(response['message']['content'])
```

**Hands-on experiment:**

```bash
# Raw Mistral (no context) — ask a scheme question
ollama run mistral "What is the eligibility for PM-KISAN?"
# → Likely gives a generic/possibly wrong answer from training data

# Then run the same query through YOUR RAG pipeline
python scripts/query.py "What is the eligibility for PM-KISAN?"
# → Should give a specific, grounded answer from your scheme docs

# The difference demonstrates WHY RAG > raw LLM for domain-specific tasks
```

---

### 5. Debugging RAG Pipelines

When answers are bad, the problem is usually in one of two places:

| Symptom                             | Root Cause        | Fix                                            |
| ----------------------------------- | ----------------- | ---------------------------------------------- |
| Answer is about the wrong scheme    | Retriever issue   | Check chunk scores, adjust top_k or chunk size |
| Answer has facts not in the context | LLM hallucinating | Strengthen system prompt grounding instruction |
| Answer is vague/incomplete          | Chunks too small  | Increase chunk size in ingestion               |
| Answer repeats itself               | Duplicate chunks  | Check for duplicate docs in ChromaDB           |
| Answer says "I don't know"          | Retriever miss    | Check if the data actually contains the answer |

**Hands-on:** When a test query gives a bad answer, print the retrieved chunks first
to determine if the problem is retrieval or generation.

---

### Summary: Key Takeaways for Today

1. **RAG = Retrieve first, then Generate** — the retriever determines answer quality
   more than the LLM does
2. **Same embedding model** for ingestion and query — mismatch = broken search
3. **Grounding instructions** in the system prompt prevent hallucination
4. **Local LLMs** (Mistral via Ollama) are good enough for MVP — free and private
5. **Debug by checking retrieval first** — if wrong chunks come back, fix the
   retriever before blaming the LLM
