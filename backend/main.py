import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PyPDF2 import PdfReader
import chromadb
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

RAG_COLLECTION= "rag_documents"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "models/gemini-3.1-flash-lite"
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 3
BATCH_SIZE = 32

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name=RAG_COLLECTION)

app = FastAPI(title="RAG Chat Backend", description="PDF-based retrieval augmented generation chat using Google Gemini.")

class ChatRequest(BaseModel):
    message: str
    top_k: int = DEFAULT_TOP_K

class ChatResponse(BaseModel):
    reply: str
    source_documents: List[dict]


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def reset_collection():
    global collection
    try:
        chroma_client.delete_collection(RAG_COLLECTION)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=RAG_COLLECTION)
    return collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
            embeddings.extend([item.values for item in response.embeddings])
        except Exception as e:
            raise RuntimeError(f"Embedding failed for batch {i//BATCH_SIZE}: {str(e)}")
    return embeddings


def index_all_pdfs() -> int:
    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        reset_collection()
        return 0

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[dict] = []

    for pdf_path in pdf_files:
        text = extract_text(pdf_path)
        if not text:
            continue

        for chunk_index, chunk in enumerate(chunk_text(text)):
            ids.append(f"{pdf_path.name}-{chunk_index}")
            docs.append(chunk)
            metadatas.append({"source": pdf_path.name, "chunk": chunk_index})

    if not docs:
        reset_collection()
        return 0

    embeddings = embed_texts(docs)
    reset_collection()
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    return len(docs)


def _query_to_list(result, key: str):
    if isinstance(result, dict):
        return result.get(key, [])
    if hasattr(result, key):
        return getattr(result, key)
    try:
        return result[key]
    except Exception:
        return []


def get_search_results(query: str, top_k: int = DEFAULT_TOP_K) -> List[dict]:
    if collection.count() == 0:
        raise RuntimeError("No documents indexed. Upload a PDF first.")

    query_embedding = client.models.embed_content(model=EMBEDDING_MODEL, contents=[query])
    query_vector = query_embedding.embeddings[0].values
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Result is a dict with lists as values
    documents = result.get("documents", [[]])[0] if result.get("documents") else []
    metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
    distances = result.get("distances", [[]])[0] if result.get("distances") else []

    results: List[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        results.append({"document": doc, "metadata": meta or {}, "distance": float(dist)})
    return results


def format_contexts(results: List[dict]) -> str:
    formatted: List[str] = []
    for item in results:
        source = item["metadata"].get("source", "unknown")
        chunk_id = item["metadata"].get("chunk", -1)
        formatted.append(f"Source: {source} (chunk {chunk_id})\n{item['document']}")
    return "\n\n---\n\n".join(formatted)


def generate_answer(question: str, contexts: str) -> str:
    try:
        prompt = (
            "Use the provided document context to answer the user question.\n"
            "If the answer is not contained in the context, say that you do not know.\n\n"
            f"Context:\n{contexts}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[prompt],
            config={"temperature": 0.2, "max_output_tokens": 512},
        )

        if not response.candidates:
            return "No response generated."

        candidate = response.candidates[0]
        answer = ""
        if hasattr(candidate, "content") and candidate.content:
            for part in getattr(candidate.content, "parts", []):
                answer += getattr(part, "text", "") or ""
        return answer.strip() or "No text response."
    except Exception as e:
        raise RuntimeError(f"Answer generation failed: {str(e)}")


@app.on_event("startup")
def startup_indexer():
    index_all_pdfs()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
        destination = UPLOAD_DIR / Path(file.filename).name
        content = await file.read()
        
        with destination.open("wb") as buffer:
            buffer.write(content)
        
        indexed_chunks = index_all_pdfs()
        return {"filename": file.filename, "indexed_chunks": indexed_chunks}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        if collection.count() == 0:
            raise HTTPException(status_code=404, detail="No documents indexed. Upload a PDF first.")
        
        results = get_search_results(request.message, top_k=request.top_k)
        
        if not results:
            raise HTTPException(status_code=404, detail="No matching documents found.")
        
        contexts = format_contexts(results)
        answer = generate_answer(request.message, contexts)
        return ChatResponse(reply=answer, source_documents=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
