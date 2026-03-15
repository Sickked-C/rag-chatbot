from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from decouple import config
import os
import shutil

app = FastAPI(title="RAG Chatbot API")

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
GROQ_API_KEY = config('GROQ_API_KEY')
UPLOAD_DIR = "uploaded_docs"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_state = {"retriever": None}
groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────
# Build RAG pipeline
# ─────────────────────────────────────────
def build_rag_chain(file_path: str):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[DEBUG] Loaded {len(documents)} pages")

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"[DEBUG] Split into {len(chunks)} chunks")

    # 3. Embed locally với HuggingFace (không cần API key)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    _state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"[DEBUG] Retriever ready!")

    return len(chunks)

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "RAG Chatbot API — Groq + HuggingFace"}

@app.get("/health")
def health():
    return {"status": "ok", "document_loaded": _state["retriever"] is not None}

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
    if _state["retriever"] is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first."
        )

    try:
        # Retrieve relevant chunks
        docs = _state["retriever"].invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Call Groq LLM
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based only on the provided context."
                },
                {
                    "role": "user",
                    "content": f"""Context:
{context}

Question: {question}

Answer based only on the context above. If the answer is not in the context, say "I don't have enough information." """
                }
            ],
            temperature=0.3
        )

        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "page": doc.metadata.get("page", 0) + 1,
                    "content": doc.page_content[:200]
                }
                for doc in docs
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))