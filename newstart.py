from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Configure Gemini
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 2. Load PDF
loader = PyPDFLoader("/Users/nikhil/Downloads/Learning to Love Data Science.pdf")
docs = loader.load()
for i, d in enumerate(docs):
    d.metadata["page"] = i + 1
text = " ".join([d.page_content for d in docs])
print(f"Loaded {len(docs)} pages, total {len(text)} characters.")

# 3. Chunk text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks.")

# 4. Build embeddings + vector store
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
db = FAISS.from_texts(chunks, emb)
db.save_local("vectorstore")

# 5. Reload for retrieval
db = FAISS.load_local("vectorstore", emb, allow_dangerous_deserialization=True)

# 6. Query and retrieve context
query = "Tell me about the author."
results = db.similarity_search(query, k=3)
context = "\n\n".join([r.page_content for r in results])

# 7. Send to Gemini for answer generation
prompt = f"""
You are a helpful assistant.

Use the following context from the PDF to answer the question clearly.

Context:
{context}

Question: {query}
Answer:
"""
model = genai.GenerativeModel("models/gemini-2.5-flash")
response = model.generate_content(prompt)
print("\nGemini Response:\n", response.text)
