🤖 Chaty_PDFs – The Future of Document Interaction

Chaty_PDFs is an AI-powered chatbot that transforms the way you interact with PDF documents. Instead of scrolling through hundreds of pages, simply upload your PDF and start chatting with it.

Powered by FastAPI, LangChain, LLaMA, HuggingFace embeddings, and ChromaDB, the chatbot extracts knowledge from PDFs, stores it intelligently, and delivers human-like answers with context awareness.

🚀 Why Chaty_PDFs is Different
Traditional PDF tools only let you search for keywords. This fails when you need:
Context across multiple sections.
Summaries of complex topics.
Instant answers from large documents.

Chaty_PDFs solves these problems:
✅ Understands context – grasps meaning, not just words.
✅ Answers with precision – retrieves exact sections, summarizes, and explains.
✅ Scales with large PDFs – embeddings + vector databases handle thousands of pages.
✅ Feels like chat – natural, real-time responses.
✅ Future-proof – designed to extend beyond text (images, graphs, code).

👉 This isn’t just a chatbot. It’s a personal AI knowledge assistant.

🏗️ Folder Structure
Chaty_PDFs/
│── backend/  
│   ├── main.py               # FastAPI backend (core API for chat & PDF upload)  
│   ├── extractor.py          # PDF → Text → Embeddings → ChromaDB  
│   ├── requirements.txt      # Python dependencies  
│   ├── README.md             # Backend-specific docs  
│   │
│   │── Pdfs/
│   │
│   ├── static/               # Frontend served by FastAPI  
│   │   ├── index.html        # Chat interface (Tailwind CSS + JS)  
│   │
│   ├── pdf_files/            # Temporary storage for uploaded PDFs  
│   ├── chroma_db/            # Vector database (stores embeddings)  
│   └── logs/                 # Debugging & query logs  
│
│── .gitignore                # Ignore cache, venv, logs, DB files  
│── README.md                 # Project-level documentation  

🔍 Explanation of Key Folders
backend: This is the core engine of Chaty_PDFs. It contains the FastAPI backend, which runs APIs for chatting with documents, handling search queries, and managing PDF uploads and processing.

static: The frontend lives here. It’s a clean, responsive web interface built with Tailwind CSS and JavaScript, allowing users to upload PDFs and interact with the chatbot in real time.

pdf_files: This folder acts as temporary storage for uploaded PDFs. Once a file is processed and converted into embeddings, it can be safely referenced or cleared as needed.

chroma_db: This is the knowledge brain of the system. All embeddings, metadata, and semantic representations of documents are stored here, enabling efficient semantic search and retrieval.

logs: This directory stores logs, including debugging details, query traces, and errors. It’s especially useful for monitoring system performance and troubleshooting issues.

⚙️ How It Works (Step-by-Step Technical Flow)
The magic happens in four stages:

1. Document Ingestion
The document ingestion process begins when a user uploads a PDF through the frontend. The system uses PyMuPDF to extract raw text page by page, ensuring that the structure and readability of the document are preserved. Since feeding an entire PDF into the model would be inefficient and imprecise, the extracted text is then split into semantic chunks of about 650 characters with a small overlap. This chunking ensures that embeddings remain contextually meaningful, avoids cutting off important sentences, and allows queries to precisely target specific sections of the document for accurate retrieval later.

2. Embedding Generation
In the embedding generation step, each text chunk is transformed into a 384-dimensional vector using the HuggingFace model sentence-transformers/all-MiniLM-L6-v2. These embeddings capture the semantic meaning of the text rather than just keyword matches, enabling context-aware retrieval. To maintain consistency, embeddings are normalized, ensuring stable and accurate similarity scoring. As a result, semantically similar sentences like “AI is a branch of ML” and “Machine learning includes AI” are positioned closely in the vector space, allowing the system to recognize them as related concepts even if the exact wording differs.

3. Knowledge Storage in ChromaDB
In the knowledge storage phase, each embedding is saved in ChromaDB along with its associated metadata such as book name, page number, and chunk ID.
Example entry:
{
  "id": "uuid1234",
  "book_name": "DeepLearningIntro",
  "page_number": 5,
  "content": "Deep learning is a subset of ML using neural networks..."
}
This structured storage ensures that every vector can be traced back to its exact location in the document. Since the database is persisted locally in /chroma_db/, previously processed documents do not need to be re-embedded, making the system more efficient and scalable.

4. Semantic Search + Retrieval-Augmented Generation (RAG)
In the semantic search and Retrieval-Augmented Generation (RAG) phase, a user’s query is first embedded into a vector using the same HuggingFace model to ensure consistency. This query vector is then compared against stored embeddings in ChromaDB using cosine similarity, which identifies the most semantically relevant chunks of text. The retrieved context is combined with the original query and fed into the LLaMA model (running locally as a GGUF quantized version). Leveraging both the user’s intent and the contextual knowledge, LLaMA generates a natural, accurate, and context-aware response, allowing the chatbot to deliver answers that go beyond simple keyword search.

🔎 How Text Embedding Works in Detail

The process begins with text extraction and chunking. Once a PDF is uploaded, its raw text is extracted using PyMuPDF, a reliable library that can handle even complex PDFs with images and formatting. Since feeding an entire PDF into an embedding model would be inefficient and inaccurate, the text is broken into manageable semantic chunks using LangChain’s RecursiveCharacterTextSplitter. By default, each chunk is around 650 characters with a 50-character overlap to preserve context across boundaries. This ensures that every chunk is meaningful, self-contained, and ready for embedding.

Next comes the embedding step, where each chunk of text is transformed into a dense vector representation using a HuggingFace embedding model (e.g., sentence-transformers/all-MiniLM-L6-v2). Unlike keyword search, embeddings capture the semantic meaning of the text — so two chunks about “neural networks” and “deep learning” would be close together in vector space, even if they don’t share exact words. This allows the system to truly "understand" the content instead of just matching strings.

Once generated, these embeddings are stored in ChromaDB, a specialized vector database. Along with the embedding vectors, metadata such as book/document name, page number, and unique chunk IDs is also saved. This enables efficient retrieval later. Importantly, one PDF page often becomes multiple chunks, each stored as a separate searchable entry in ChromaDB. This design ensures that even very large documents remain searchable at fine granularity.

Finally, when a user submits a query, the same embedding model encodes the question into a vector. ChromaDB then performs a semantic similarity search — comparing the query vector with all stored vectors using cosine similarity. The database returns the top-matching chunks along with their metadata (document name, page number, confidence score). These chunks are then passed to the LLaMA model, which uses them as context to generate a natural, context-aware answer, ensuring that results are accurate, meaningful, and directly linked to the source text.
📌 Pipeline Summary:

PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Semantic Search → LLaMA Answer

📸 Workflow Diagram
[ PDF Upload ] 
       ↓
[ Text Extraction → Chunking ] 
       ↓
[ HuggingFace Embeddings ] 
       ↓
[ ChromaDB Vector Store ]
       ↓
User Question → [ Embedding ] → [ Semantic Search ] → [ LLaMA Model ] → Answer

🔧 Tech Stack

Backend: FastAPI, LangChain, HuggingFace Embeddings, LLaMA.cpp, ChromaDB
Frontend: Tailwind CSS + Vanilla JS (dark mode, responsive UI)
PDF Parsing: PyMuPDF (fitz)
Infrastructure: Local ChromaDB, caching, modular design
Other Tools: tqdm (progress bars), logging (debug & monitor queries)

🖥️ Getting Started
Clone the Repository
git clone https://github.com/your-username/Chaty_PDFs.git
cd Chaty_PDFs/backend

Install Dependencies
pip install -r requirements.txt

Build the Knowledge Base
Run the extractor to embed your PDFs into ChromaDB:
Then
python extractor.py

Run the App
uvicorn main:app --reload

Open in Browser
👉 http://127.0.0.1:8000

💡 Example Use Cases

📚 Education – Chat with textbooks, research papers, lecture slides.
🏢 Enterprise – Query company policies, contracts, compliance docs.
⚖️ Legal – Extract insights from case laws, agreements, regulations.
🏥 Healthcare – Summarize clinical research, guidelines, and medical papers.
📰 Research & Media – Digest long reports and archives in minutes.

✅ Roadmap / Future Enhancements

🔍 Support for multi-PDF queries.
🎙️ Voice-enabled queries.
🌐 One-click cloud deployment (AWS, Railway, Heroku).
🧠 Multi-modal input (text + images + graphs).
👥 Role-based authentication and saved conversations.

📜 License
MIT License — free to use, modify, and share.

🙌 Acknowledgments
LangChain
LLaMA.cpp
ChromaDB
FastAPI
PyMuPDF

✨ Chaty_PDFs is not just code — it’s a step toward the future of human-AI collaboration with knowledge.