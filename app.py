import os
import nltk
import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

api_key = os.getenv("API_KEY")
gemini = genai.Client(api_key=api_key)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EMBED_MODEL = "models/gemini-embedding-001"

# --- Embedding via Gemini ---

def get_embedding(text: str) -> list[float]:
    response = gemini.models.embed_content(model=EMBED_MODEL, contents=[text])
    return list(response.embeddings[0].values)


class GeminiEmbedWrapper(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [get_embedding(text) for text in input]


embed_fn = GeminiEmbedWrapper()
chroma_client = chromadb.PersistentClient(path="chroma_store")


# --- Helpers ---

def get_collection(pdf_name: str):
    import re
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", pdf_name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")
    if not safe_name[0].isalnum():
        safe_name = "c" + safe_name
    if not safe_name[-1].isalnum():
        safe_name = safe_name + "0"
    return chroma_client.get_or_create_collection(
        name=safe_name, embedding_function=embed_fn
    )


def extract_text(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text, chunk_size=5, chunk_overlap=1):
    sentences = nltk.sent_tokenize(text)
    chunks, i = [], 0
    while i < len(sentences):
        chunks.append(" ".join(sentences[i:i + chunk_size]))
        i += chunk_size - chunk_overlap
    return chunks


def ingest(pdf_name: str, filepath: str):
    collection = get_collection(pdf_name)
    if collection.count() > 0:
        return
    text = extract_text(filepath)
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.upsert(
            ids=[f"{pdf_name}_chunk{i+1}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": pdf_name, "chunk_index": i + 1}]
        )


# --- Query Rewriting ---

def rewrite_query(question: str, chat) -> list[str]:
    prompt = (
        f"Generate 3 different search queries to retrieve relevant information for: '{question}'. "
        "Return only the 3 queries, one per line, no numbering."
    )
    variants = [q.strip() for q in chat.send_message(prompt).text.strip().split("\n") if q.strip()]
    return [question] + variants[:3]


# --- Reranking via Gemini ---

def rerank(question: str, chunks: list[dict], chat, top_k=3) -> list[dict]:
    if not chunks:
        return []
    chunk_list = "\n".join(f"{i+1}. {c['text'][:300]}" for i, c in enumerate(chunks))
    prompt = (
        f"Given the question: '{question}'\n\n"
        f"Rank these chunks by relevance (most relevant first). "
        f"Return only the numbers in order, comma-separated.\n\n{chunk_list}"
    )
    response = chat.send_message(prompt).text.strip()
    try:
        order = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
        ranked = [chunks[i] for i in order if i < len(chunks)]
        # append any missed chunks
        seen = set(order)
        ranked += [chunks[i] for i in range(len(chunks)) if i not in seen]
        return ranked[:top_k]
    except Exception:
        return chunks[:top_k]


# --- Retrieval ---

def query_documents(question: str, pdf_name: str, chat, n_results=5, top_k=3):
    collection = get_collection(pdf_name)
    queries = rewrite_query(question, chat)

    seen, all_chunks = set(), []
    for q in queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if doc not in seen:
                seen.add(doc)
                all_chunks.append({"text": doc, "source": meta.get("source", "")})

    return rerank(question, all_chunks, chat, top_k)


# --- Response Generation ---

def generate_response(question: str, chunks: list, chat) -> str:
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following retrieved context "
        "to answer the question. If you don't know the answer, say so. Keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    return chat.send_message(prompt).text


# --- FastAPI ---

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

chat_sessions: dict[str, object] = {}


class QuestionRequest(BaseModel):
    pdf_name: str
    question: str


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    ingest(file.filename, filepath)
    return {"message": f"'{file.filename}' uploaded and ingested.", "pdf_name": file.filename}


@app.get("/pdfs")
def list_pdfs():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    return {"pdfs": files}


@app.post("/ask")
def ask_question(req: QuestionRequest):
    filepath = os.path.join(UPLOAD_DIR, req.pdf_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="PDF not found.")
    if req.pdf_name not in chat_sessions:
        chat_sessions[req.pdf_name] = gemini.chats.create(model="models/gemini-3.5-flash")
    chat = chat_sessions[req.pdf_name]
    chunks = query_documents(req.question, req.pdf_name, chat)
    answer = generate_response(req.question, chunks, chat)
    return {"answer": answer}
