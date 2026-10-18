import os
import re
import json
import sqlite3
import nltk
import chromadb
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

load_dotenv()

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

api_key = os.getenv("API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "changeme-use-a-long-random-string")
JWT_EXPIRE_HOURS = 24

gemini = genai.Client(api_key=api_key)
bearer = HTTPBearer()

BASE_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(BASE_DIR, "users.json")
DB_FILE = os.path.join(BASE_DIR, "chat_history.db")
EMBED_MODEL = "models/gemini-embedding-001"

chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_store"))


# --- SQLite setup ---

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            pdf_name TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_message(username: str, pdf_name: str, role: str, message: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO chat_history (username, pdf_name, role, message, timestamp) VALUES (?, ?, ?, ?, ?)",
        (username, pdf_name, role, message, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

def load_history(username: str, pdf_name: str) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT role, message, timestamp FROM chat_history WHERE username=? AND pdf_name=? ORDER BY id ASC",
        (username, pdf_name)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "message": r["message"], "timestamp": r["timestamp"]} for r in rows]

def delete_history(username: str, pdf_name: str):
    conn = get_db()
    conn.execute("DELETE FROM chat_history WHERE username=? AND pdf_name=?", (username, pdf_name))
    conn.commit()
    conn.close()


# --- User store ---

def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


# --- JWT ---

def create_token(username: str) -> str:
    payload = {"sub": username, "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


# --- Per-user paths ---

def user_upload_dir(username: str) -> str:
    path = os.path.join(BASE_DIR, "uploads", username)
    os.makedirs(path, exist_ok=True)
    return path

def get_collection_name(username: str, pdf_name: str) -> str:
    raw = f"{username}_{pdf_name}"
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe[0].isalnum():
        safe = "c" + safe
    if not safe[-1].isalnum():
        safe = safe + "0"
    return safe[:512]

def get_collection(username: str, pdf_name: str):
    return chroma_client.get_or_create_collection(
        name=get_collection_name(username, pdf_name),
        embedding_function=GeminiEmbedWrapper()
    )


# --- Embedding ---

class GeminiEmbedWrapper(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [list(gemini.models.embed_content(model=EMBED_MODEL, contents=[t]).embeddings[0].values) for t in input]


# --- Ingestion ---

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

def ingest(username: str, pdf_name: str, filepath: str):
    collection = get_collection(username, pdf_name)
    if collection.count() > 0:
        return
    text = extract_text(filepath)
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        embedding = list(gemini.models.embed_content(model=EMBED_MODEL, contents=[chunk]).embeddings[0].values)
        collection.upsert(
            ids=[f"{pdf_name}_chunk{i+1}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{"source": pdf_name, "chunk_index": i + 1}]
        )


# --- RAG ---

def rewrite_query(question: str, chat) -> list[str]:
    prompt = (
        f"Generate 3 different search queries to retrieve relevant information for: '{question}'. "
        "Return only the 3 queries, one per line, no numbering."
    )
    variants = [q.strip() for q in chat.send_message(prompt).text.strip().split("\n") if q.strip()]
    return [question] + variants[:3]

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
        seen = set(order)
        ranked += [chunks[i] for i in range(len(chunks)) if i not in seen]
        return ranked[:top_k]
    except Exception:
        return chunks[:top_k]

def query_documents(question: str, username: str, pdf_name: str, chat, n_results=5, top_k=3):
    collection = get_collection(username, pdf_name)
    queries = rewrite_query(question, chat)
    seen, all_chunks = set(), []
    for q in queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if doc not in seen:
                seen.add(doc)
                all_chunks.append({"text": doc, "source": meta.get("source", "")})
    return rerank(question, all_chunks, chat, top_k)

def generate_response(question: str, chunks: list, chat) -> str:
    context = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following retrieved context "
        "to answer the question. If you don't know the answer, say so. Keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    return chat.send_message(prompt).text


# --- Restore chat session from DB ---

def restore_session(username: str, pdf_name: str) -> object:
    chat = gemini.chats.create(model="models/gemini-3.5-flash")
    history = load_history(username, pdf_name)
    for entry in history:
        if entry["role"] == "user":
            chat.send_message(entry["message"])
    return chat


# --- FastAPI ---

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

chat_sessions: dict[str, object] = {}


class AuthRequest(BaseModel):
    username: str
    password: str

class QuestionRequest(BaseModel):
    pdf_name: str
    question: str


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/register")
def register(req: AuthRequest):
    users = load_users()
    if req.username in users:
        raise HTTPException(status_code=400, detail="Username already exists.")
    users[req.username] = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    save_users(users)
    return {"token": create_token(req.username)}


@app.post("/login")
def login(req: AuthRequest):
    users = load_users()
    if req.username not in users or not bcrypt.checkpw(req.password.encode(), users[req.username].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"token": create_token(req.username)}


@app.post("/logout")
def logout(username: str = Depends(verify_token)):
    keys = [k for k in chat_sessions if k.startswith(f"{username}:")]
    for k in keys:
        del chat_sessions[k]
    return {"message": "Logged out."}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), username: str = Depends(verify_token)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    filepath = os.path.join(user_upload_dir(username), file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    ingest(username, file.filename, filepath)
    return {"message": f"'{file.filename}' uploaded and ingested.", "pdf_name": file.filename}


@app.get("/pdfs")
def list_pdfs(username: str = Depends(verify_token)):
    files = [f for f in os.listdir(user_upload_dir(username)) if f.endswith(".pdf")]
    return {"pdfs": files}


@app.delete("/pdfs/{pdf_name}")
def delete_pdf(pdf_name: str, username: str = Depends(verify_token)):
    filepath = os.path.join(user_upload_dir(username), pdf_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="PDF not found.")
    os.remove(filepath)
    try:
        chroma_client.delete_collection(get_collection_name(username, pdf_name))
    except Exception:
        pass
    chat_sessions.pop(f"{username}:{pdf_name}", None)
    delete_history(username, pdf_name)
    return {"message": f"'{pdf_name}' deleted."}


@app.get("/history/{pdf_name}")
def get_history(pdf_name: str, username: str = Depends(verify_token)):
    return {"history": load_history(username, pdf_name)}


@app.post("/ask")
def ask_question(req: QuestionRequest, username: str = Depends(verify_token)):
    filepath = os.path.join(user_upload_dir(username), req.pdf_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="PDF not found.")
    session_key = f"{username}:{req.pdf_name}"
    if session_key not in chat_sessions:
        chat_sessions[session_key] = restore_session(username, req.pdf_name)
    chat = chat_sessions[session_key]
    chunks = query_documents(req.question, username, req.pdf_name, chat)
    answer = generate_response(req.question, chunks, chat)
    save_message(username, req.pdf_name, "user", req.question)
    save_message(username, req.pdf_name, "bot", answer)
    return {"answer": answer}
