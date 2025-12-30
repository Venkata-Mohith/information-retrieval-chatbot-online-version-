import streamlit as st
import os
from dotenv import load_dotenv
import warnings
import time
import logging

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")
load_dotenv()
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Embeddings
# -----------------------------
class LocalSentenceTransformerEmbeddings:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        start = time.time()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info(f"Embeddings loaded in {time.time() - start:.2f}s")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text]).tolist()[0]

# -----------------------------
# Load PDFs
# -----------------------------
def load_pdfs():
    pdf_dir = "missile_pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    return [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ]

# -----------------------------
# Vector Store
# -----------------------------
@st.cache_resource
def setup_vector_store():
    embeddings = LocalSentenceTransformerEmbeddings()
    persist_dir = "chroma_db"

    if os.path.exists(persist_dir):
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        docs = []
        for pdf in load_pdfs():
            loader = PyMuPDFLoader(pdf)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=persist_dir
        )
        db.persist()

    return db.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM
# -----------------------------
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY missing in Streamlit secrets")
        st.stop()

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

# -----------------------------
# Prompt
# -----------------------------
PROMPT = PromptTemplate(
    template="""
Use ONLY the context below to answer.
If not found, say:
"I can only answer based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# -----------------------------
# QA Chain
# -----------------------------
@st.cache_resource
def init_qa():
    retriever = setup_vector_store()
    llm = get_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

# -----------------------------
# UI
# -----------------------------
st.title("🚀 DRDO Missile Systems Chatbot")

qa = init_qa()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a missile-related question"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa.run(user_input)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
