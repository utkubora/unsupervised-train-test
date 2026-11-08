"""
Local RAG pipeline to reduce hallucinations.

Setup:
    pip install ollama chromadb pypdf
    ollama pull nomic-embed-text      # embedding model
    ollama pull llama3.1              # or any chat model you use

Usage:
    python rag.py ingest ./docs       # index your .txt / .md / .pdf files
    python rag.py ask "your question"
    python rag.py chat                # interactive mode
"""

import sys
import hashlib
from pathlib import Path

import chromadb
import ollama

# ---------------- Config ----------------
CHAT_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"
DB_PATH = "./rag_db"
COLLECTION = "docs"
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap keeps sentences from being cut in half
TOP_K = 5                 # chunks retrieved per question
MAX_DISTANCE = 0.55       # cosine distance cutoff; above this = "not relevant"

SYSTEM_PROMPT = """You are a careful assistant that answers ONLY from the provided context.
Rules:
1. Use only facts stated in the context. Do not use outside knowledge.
2. After each claim, cite the source in brackets, e.g. [manual.pdf #3].
3. If the context does not contain the answer, reply exactly:
   "I don't know based on the provided documents."
4. Never guess, invent numbers, names, or dates."""

# ---------------- Storage ----------------
client = chromadb.PersistentClient(path=DB_PATH)
col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def read_file(p: Path) -> str:
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(p).pages)
    return p.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        # try to end on a sentence boundary
        cut = text.rfind(". ", start, end)
        if cut > start + CHUNK_SIZE // 2:
            end = cut + 1
        chunks.append(text[start:end].strip())
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if len(c) > 30]


def ingest(folder: str):
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    for p in files:
        pieces = chunk(read_file(p))
        if not pieces:
            continue
        ids = [hashlib.md5(f"{p}:{i}:{c}".encode()).hexdigest() for i, c in enumerate(pieces)]
        for i in range(0, len(pieces), 64):  # batch embeddings
            batch = pieces[i:i + 64]
            col.upsert(
                ids=ids[i:i + 64],
                documents=batch,
                embeddings=embed(batch),
                metadatas=[{"source": p.name, "chunk": i + j} for j in range(len(batch))],
            )
        print(f"Indexed {p.name}: {len(pieces)} chunks")
    print(f"Done. Total chunks in DB: {col.count()}")


def retrieve(question: str):
    res = col.query(query_embeddings=embed([question]), n_results=TOP_K)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= MAX_DISTANCE:  # drop weak matches so the model isn't fed noise
            hits.append((doc, meta, dist))
    return hits


def ask(question: str) -> str:
    hits = retrieve(question)
    if not hits:
        # Refuse BEFORE calling the model: no context = no chance to hallucinate
        return "I don't know based on the provided documents."

    context = "\n\n".join(f"[{m['source']} #{m['chunk']}]\n{d}" for d, m, _ in hits)
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0},  # deterministic, less creative = fewer fabrications
    )
    answer = resp["message"]["content"]
    sources = sorted({f"{m['source']} #{m['chunk']} (dist {dist:.2f})" for _, m, dist in hits})
    return answer + "\n\nRetrieved:\n  " + "\n  ".join(sources)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == "ingest":
        ingest(sys.argv[2] if len(sys.argv) > 2 else "./docs")
    elif sys.argv[1] == "ask":
        print(ask(" ".join(sys.argv[2:])))
    elif sys.argv[1] == "chat":
        while (q := input("\n> ").strip()) not in {"", "exit", "quit"}:
            print(ask(q))
