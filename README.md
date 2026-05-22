# 📚 Local RAG Chat Application

## 🚀 Project Overview

This project provides a complete **Retrieval-Augmented Generation (RAG)** pipeline that bridges the gap between your local documents and a Large Language Model (LLM). By utilizing **Ollama** as the local inference engine and **FAISS** as the vector store, the application ensures that your sensitive data remains private and resides entirely on your local infrastructure.

---

# 🛠️ Technical Highlights

The system is engineered to handle varied document types and complex parsing requirements.

## 📄 Intelligent Parsing
The `rag.py` module includes:

- Specific logic for handling **multi-column PDF layouts** using `pdfplumber`
- Robust fallback mechanisms for `.docx` files
- Efficient text extraction and preprocessing pipelines

---

## 🔍 Vector Search & Optimization

The application uses:

- **`nomic-embed-text`** for generating high-quality semantic embeddings
- **`faiss-cpu`** for fast and efficient similarity search across thousands of document chunks

This enables low-latency and accurate retrieval of contextually relevant information.

---

## ⚡ Performance-Oriented Features

### 💾 Disk Caching
Uses Python’s `shelve` module with **MD5-based hashing** to cache embeddings and prevent redundant processing of previously indexed files.

### ⚙️ Parallel Processing
Employs `ThreadPoolExecutor` to accelerate embedding generation during the indexing phase.

### 🌐 Streamlit Integration
Provides a responsive chat interface that:

- Tracks conversation state
- Maintains persistent chat history using `sqlite3`
- Supports interactive document-based querying

---

# 🔄 Workflow Diagram

The application follows a systematic Retrieval-Augmented Generation workflow.

## 1️⃣ Ingestion
Files are:

- Loaded
- Parsed
- Split into chunks using:

```env
CHUNK_SIZE
CHUNK_OVERLAP
```

defined in the `.env` configuration file.

---

## 2️⃣ Embedding
Each document chunk is converted into a vector representation using the embedding model and stored inside a **FAISS index**.

---

## 3️⃣ Retrieval
When a user submits a query:

- The query is transformed into an embedding vector
- The system retrieves the `RETRIEVE_TOP_K` most relevant chunks from FAISS

---

## 4️⃣ Generation
The retrieved chunks are injected into the prompt as contextual information. The **`qwen2:1.5b`** model then generates an accurate, source-grounded response.

---

# 🧰 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Engine | Ollama |
| Embedding Model | nomic-embed-text |
| Vector Database | FAISS |
| Database | SQLite3 |
| PDF Parsing | pdfplumber |
| Concurrency | ThreadPoolExecutor |
| Caching | shelve + MD5 |

---

# 📂 Core Features

- 📄 Multi-format document ingestion
- 🔍 Semantic search using embeddings
- ⚡ Fast retrieval with FAISS
- 💬 Interactive chatbot interface
- 🧠 Local LLM inference with Ollama
- 🔒 Fully offline and privacy-focused architecture
- 💾 Persistent chat history
- 🚀 Optimized embedding pipeline

---

# 📌 Environment Configuration

Example `.env` configuration:

```env
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
GENERATE_MODEL=qwen2:1.5b

# RAG settings
RETRIEVE_TOP_K=5
CHUNK_SIZE=2000
CHUNK_OVERLAP=200

# Generation settings
TEMPERATURE=0.3

# App settings
APP_TITLE=RAG Chat Application
MAX_FILES=10
```

---

# 🎯 Use Cases

- Personal knowledge assistant
- Research document querying
- Offline enterprise document search
- Secure local AI chatbot
- Academic and technical document analysis

---
