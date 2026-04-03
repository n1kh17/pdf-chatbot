#  Chat with Your PDF

A chatbot that reads a PDF and answers questions based on its content — built to understand how **Retrieval-Augmented Generation (RAG)** works in practice, not just theory.

---

## 🔍 How It Works

1. **Upload** a PDF
2. **Chunk** it into smaller pieces
3. **Embed** those chunks using HuggingFace embeddings
4. **Store** them in a FAISS vector database
5. **Ask a question** → it finds the most relevant chunks → sends them to Gemini → returns an answer grounded in the document

> Instead of the model guessing, it answers directly from your document.

---

##  Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Pipeline | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| Generation | Google Gemini |
| Language | Python |

---

##  Getting Started

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 5. Run the app

```bash
streamlit run chatui.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

##  Limitations

- Works with **one PDF at a time**
- **No memory** between sessions
- Answers may be off if retrieval is weak
- No advanced guardrails yet
