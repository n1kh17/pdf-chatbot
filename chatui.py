import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import shutil

# -----------------------------
# 1️⃣ Setup
# -----------------------------
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📘")
st.title("📘 AI Knowledge Assistant (Gemini + FAISS)")
st.caption("Upload a PDF and chat with its content.")

# -----------------------------
# 2️⃣ Initialize model + embedding
# -----------------------------
model = genai.GenerativeModel("models/gemini-2.5-flash")

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# Persistent FAISS store folder
VECTOR_DIR = "vectorstore"
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)

# Session state for DB and chat
if "db" not in st.session_state:
    st.session_state.db = None
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# 3️⃣ File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save PDF temporarily
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = " ".join([d.page_content for d in docs])
    st.info(f"Loaded {len(docs)} pages, {len(text)} characters")

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    st.success(f"Created {len(chunks)} text chunks.")

    # Build FAISS index
    with st.spinner("Building vector store..."):
        db = FAISS.from_texts(chunks, emb)
        db.save_local(VECTOR_DIR)
        st.session_state.db = db
    shutil.rmtree(temp_dir)
    st.success("✅ Vector store created successfully!")

# -----------------------------
# 4️⃣ Load existing index (if no upload)
# -----------------------------
if not uploaded_file and os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
    try:
        st.session_state.db = FAISS.load_local(
            VECTOR_DIR, emb, allow_dangerous_deserialization=True
        )
        st.caption("Using existing vector store.")
    except Exception as e:
        st.error(f"Error loading vector store: {e}")

# -----------------------------
# 5️⃣ Chat Interface
# -----------------------------
if st.session_state.db:
    user_query = st.chat_input("Ask something about the document...")

    if user_query:
        st.chat_message("user").markdown(user_query)

        # Retrieve top chunks
        results = st.session_state.db.similarity_search(user_query, k=3)
        context = "\n\n".join([r.page_content for r in results])

        prompt = f"""
        You are a helpful assistant. Use the following context to answer clearly.

        Context:
        {context}

        Question: {user_query}
        Answer:
        """

        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"

        st.chat_message("assistant").markdown(answer)

        # Save to chat history
        st.session_state.history.append({"role": "user", "content": user_query})
        st.session_state.history.append({"role": "assistant", "content": answer})

    # Display full chat history
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).markdown(msg["content"])

else:
    st.info("📥 Please upload a PDF to begin chatting.")
