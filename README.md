# PDF RAG Chat

A web app that lets you upload PDFs and chat with them using Advanced RAG (Retrieval-Augmented Generation) powered by Gemini AI.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://localhost:8000`

---

## Architecture

### Ingestion Pipeline
```
PDF File
   │
   ▼
PdfReader ──► Raw Text
                │
                ▼
nltk.sent_tokenize()
splits on sentence boundaries (not fixed chars)
                │
                ▼
[S1 S2 S3 S4 S5] [S4 S5 S6 S7 S8] [S7 S8 S9 S10 S11] ...
 chunk1(5 sents)   chunk2(overlap=1)  chunk3 ...
                │
                ▼
Gemini Embedding API (gemini-embedding-001)
each chunk → 3072-dim float vector
                │
                ▼
ChromaDB (PersistentClient)
stored: id | text | embedding | metadata { source, chunk_index }
```

### Query Pipeline
```
User Question
      │
      ▼
rewrite_query() via Gemini
generates 3 query variants
      │
      ▼
[Q_original, Q_variant1, Q_variant2, Q_variant3]
      │
      ▼ (all 4 queries hit ChromaDB independently)
ChromaDB.query() x4 → top 5 each → deduplicate
      │
      ▼
rerank() via Gemini
scores and orders all chunks by relevance
      │
      ▼
Top 3 most relevant chunks
      │
      ▼
generate_response() via Gemini chat session
prompt = system instruction + [Source: file] chunks + question
      │
      ▼
Answer → User
```

### Full System
```
┌─────────────────────────────────────────────┐
│           FRONTEND (index.html)             │
│                                             │
│  [Upload PDF] ──── POST /upload             │
│  [Chat UI]    ──── POST /ask                │
│  [PDF List]   ──── GET  /pdfs               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         FASTAPI BACKEND (app.py)            │
│                                             │
│  /upload  → ingest()                        │
│               ├── extract_text()            │
│               ├── split_text()              │
│               ├── get_embedding()           │
│               └── chromadb.upsert()         │
│                                             │
│  /ask     → query_documents()               │
│               ├── rewrite_query()           │
│               ├── chromadb.query() x4       │
│               └── rerank()                 │
│           → generate_response()             │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────────┐
│   ChromaDB   │    │      Gemini API       │
│  (local      │    │                      │
│  persistent) │    │  gemini-embedding-001 │
│              │    │  gemini-3.5-flash     │
│  1 collection│    │  (embeddings,        │
│  per PDF     │    │   rewriting,         │
└──────────────┘    │   reranking,         │
                    │   answering)         │
                    └──────────────────────┘
```

---

## Naive RAG vs Advanced RAG

| Feature | Naive RAG | This Project |
|---|---|---|
| **Chunking** | Fixed 1000 chars, cuts mid-sentence | Sentence-aware, 5 sentences + 1 overlap |
| **Retrieval** | 1 query → ChromaDB | 4 queries (original + 3 rewrites) → merged |
| **Reranking** | None, top-N by vector similarity only | Gemini reranks all candidates by relevance |
| **Source tracking** | None | `[Source: filename]` metadata on every chunk |
| **Context to LLM** | Raw concatenated chunks | Structured with source attribution |
| **Query mismatch** | Fails if wording differs from document | Query rewriting bridges vocabulary gap |

### Why This Matters

**Naive RAG fails** when your question wording doesn't closely match the document wording. For example:

- Question: *"What did the CEO say about profits?"*
- Document says: *"The executive director discussed revenue growth..."*

Naive RAG misses this. Advanced RAG rewrites the query into variants like *"executive director revenue statement"*, retrieves the right chunk, then reranks to confirm relevance before sending to Gemini.

---

## Project Structure

```
rag/
├── app.py            # FastAPI backend + RAG logic
├── index.html        # Frontend UI
├── requirements.txt  # Dependencies
├── .env              # API_KEY=your_gemini_key
├── uploads/          # Uploaded PDFs stored here
└── chroma_store/     # ChromaDB persistent storage
```

## Environment Variables

Create a `.env` file:

```
API_KEY=your_gemini_api_key_here
```

## Dependencies

```
fastapi
uvicorn
python-multipart
chromadb
google-genai
pypdf
nltk
python-dotenv
```

No heavy ML models required — all embeddings and reranking are done via Gemini API, keeping RAM usage under 300MB.

---

## Deployment (Free)

1. Push to GitHub
2. Connect to [Render](https://render.com) or [Railway](https://railway.app)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port 10000`
5. Add environment variable: `API_KEY=your_gemini_key`
