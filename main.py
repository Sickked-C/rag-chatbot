from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager
from groq import Groq
from decouple import config
import os
import shutil

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
GROQ_API_KEY = config('GROQ_API_KEY')
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_state = {
    "chunks": None,
    "vectorizer": None,
    "vectors": None
}

groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Ready!")
    yield

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# ─────────────────────────────────────────
# Build RAG pipeline
# ─────────────────────────────────────────
def build_rag_chain(file_path: str):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[DEBUG] Loaded {len(documents)} pages")

    # 2. Split chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"[DEBUG] Split into {len(chunks)} chunks")

    # 3. TF-IDF vectorize
    texts = [chunk.page_content for chunk in chunks]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    _state["chunks"] = texts
    _state["vectorizer"] = vectorizer
    _state["vectors"] = vectors
    print(f"[DEBUG] TF-IDF ready!")

    return len(chunks)

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "RAG Chatbot API — TF-IDF + Groq LLaMA"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": _state["chunks"] is not None
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        num_chunks = build_rag_chain(file_path)
        return {
            "message": "PDF uploaded and indexed!",
            "filename": file.filename,
            "chunks": num_chunks
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: str):
    if _state["chunks"] is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )

    try:
        # Tìm chunks liên quan bằng TF-IDF
        question_vec = _state["vectorizer"].transform([question])
        similarities = cosine_similarity(question_vec, _state["vectors"]).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        context = "\n\n".join([_state["chunks"][i] for i in top_indices])

        # Gọi Groq LLaMA
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions based only on the provided context. If the answer is not in the context, say 'I don't have enough information.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            temperature=0.3
        )

        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": [
                {"content": _state["chunks"][i][:200]}
                for i in top_indices
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
