# KisanSathi MVP — 7-Day Sprint TODO

**Deadline: 2026-03-30 (Sunday)**
**Start: 2026-03-23 (Monday)**
**Today: 2026-03-23**

> This is the focused MVP task list. Full product roadmap is in EXECUTION_PLAN.md.
> Claude: check progress against this daily. If the user is behind schedule, flag it.

---

## Day 1-2 (Mar 23-24): Setup + Document Ingestion

- [ ] Initialize project structure (folders, .gitignore, requirements.txt)
- [ ] Initialize git repo, create GitHub repository
- [ ] Collect 10-20 government scheme PDFs into `data/schemes/`
- [ ] Write `scripts/ingest.py` — parse PDFs, chunk text (500 tokens, 50 token overlap)
- [ ] Store chunks + embeddings in ChromaDB using multilingual MiniLM
- [ ] Test: query ChromaDB manually, verify relevant chunks are returned

**Exit criteria:** Can run a query against ChromaDB and get back relevant scheme chunks.

---

## Day 3 (Mar 25): Retrieval + Generation Pipeline

- [ ] Write `src/retriever.py` — embed query, search ChromaDB, return top-5 chunks
- [ ] Set up Ollama with Mistral 7B quantized (or wire up Claude API as fallback)
- [ ] Write `src/generator.py` — build prompt with system instruction + retrieved chunks + query
- [ ] Test: end-to-end English query → relevant scheme answer in terminal

**Exit criteria:** Can type an English question in terminal and get a grounded scheme answer.

---

## Day 4 (Mar 26): Verification + Translation

- [ ] Write `src/verifier.py` — check answer grounding against source chunks (overlap scoring)
- [ ] Write `src/translator.py` — Hindi → English query translation, English → Hindi response
- [ ] Test: Hindi query → English retrieval → Hindi response
- [ ] Wire all components into a single `src/pipeline.py` function

**Exit criteria:** Hindi query in → Hindi scheme answer out, with grounding check.

---

## Day 5 (Mar 27): Telegram Bot

- [ ] Create Telegram bot via BotFather, get API token
- [ ] Write `src/bot.py` — handle incoming text messages via python-telegram-bot
- [ ] Connect bot to the RAG pipeline (`pipeline.py`)
- [ ] Test: send Hindi queries on Telegram, receive scheme summaries back

**Exit criteria:** Working Telegram bot that answers farmer queries in Hindi.

---

## Day 6 (Mar 28): Testing + Polish

- [ ] Test with 10+ diverse queries (different crops, regions, income levels, land sizes)
- [ ] Tune chunk size, top-K retrieval count, and prompt based on results
- [ ] Add error handling: no relevant scheme found, LLM timeout, malformed input
- [ ] Add user-friendly messages: greeting, help command, disclaimer

**Exit criteria:** Bot handles edge cases gracefully and gives quality answers.

---

## Day 7 (Mar 29-30): Deploy + Publish

- [ ] Write README.md — setup instructions, architecture diagram, screenshots
- [ ] Add `.env.example` with required environment variable names
- [ ] Final cleanup: remove dead code, check .gitignore covers secrets
- [ ] Push to GitHub as public repo
- [ ] Deploy bot (local machine or free-tier cloud: Railway / Render / fly.io)
- [ ] Run final end-to-end demo, take screenshots or record screen capture

**Exit criteria:** Public GitHub repo with working Telegram bot that anyone can deploy.

---

## Progress Log

| Date       | Day | Status      | Notes                                    |
| ---------- | --- | ----------- | ---------------------------------------- |
| 2026-03-23 | 1   | IN PROGRESS | Project kickoff, design doc written      |
| 2026-03-24 | 2   |             |                                          |
| 2026-03-25 | 3   |             |                                          |
| 2026-03-26 | 4   |             |                                          |
| 2026-03-27 | 5   |             |                                          |
| 2026-03-28 | 6   |             |                                          |
| 2026-03-29 | 7   |             |                                          |
| 2026-03-30 | 7   |             | DEADLINE — must be on GitHub and working |

---

## Stretch Goals (if ahead of schedule)

- [ ] Voice input via Whisper (receive Telegram voice notes)
- [ ] Multi-turn conversation (follow-up questions with context)
- [ ] Web scraper for auto-collecting scheme documents
- [ ] Response confidence score shown to user
- [ ] Basic analytics: log queries and responses for later analysis
