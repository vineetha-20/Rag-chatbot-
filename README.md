Project Overview:
This project provides a complete RAG pipeline that bridges the gap between your local documents and a Large Language Model (LLM). By utilizing Ollama as the local inference engine and FAISS as the vector store, the application ensures that your sensitive data remains private and resides entirely on your local infrastructure.
Technical Highlights:
The system is engineered to handle varied document types and complex parsing requirements:
Intelligent Parsing: The rag.py module includes specific logic for multi-column PDF layouts using pdfplumber and robust fallback mechanisms for docx files. 
Vector Search & Optimization: It uses nomic-embed-text for high-quality semantic representations and faiss-cpu to perform efficient, low-latency similarity searches across thousands of document chunks.  
Performance-Oriented:
    Disk Caching: Uses Python’s shelve module with MD5-based hashing to cache embeddings, preventing redundant processing of files you have already indexed.
    Parallelization: Employs ThreadPoolExecutor to speed up the embedding of document chunks during the indexing phase.  
Streamlit Integration: The frontend provides a responsive chat interface that tracks conversation state and manages persistent chat history via sqlite3.  
Workflow Diagram:
The application follows a systematic flow:
Ingestion: Files are loaded, parsed, and broken down into chunks defined by CHUNK_SIZE and CHUNK_OVERLAP in your .env file.  
Embedding: Each chunk is passed through the embedding model to create a vector representation, which is stored in a FAISS index.  
Retrieval: Upon a user query, the system converts the input into a vector and finds the RETRIEVE_TOP_K most relevant chunks.  
Generation: These chunks are injected as context into a prompt, which the qwen2:1.5b model uses to synthesize an accurate, source-grounded response.
