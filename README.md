# 🤖 RAG Chatbot API

A production-ready RAG (Retrieval-Augmented Generation) chatbot
that answers questions based on uploaded PDF documents.

## 🔍 How it works

```
PDF Upload → Chunk text → HuggingFace Embeddings → ChromaDB
                                                        ↓
User Question → Embed → Retrieve similar chunks → Groq LLaMA → Answer
```

## 🛠️ Tech Stack

| Component       | Technology                   |
| --------------- | ---------------------------- |
| API Framework   | FastAPI                      |
| LLM             | Groq LLaMA 3.1 8B            |
| Embeddings      | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | ChromaDB                     |
| PDF Processing  | LangChain + PyPDF            |

## 📦 Installation

```bash
git clone https://github.com/Sickked-C/rag-chatbot.git
cd rag-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Điền GROQ_API_KEY vào .env
python main.py
```

## 🚀 Usage

### 1. Upload PDF

```bash
POST /upload
```

### 2. Ask questions

```bash
POST /ask?question=Your question here
```

### 3. API Docs

Visit `http://localhost:8000/docs`

## 📄 License

MIT
