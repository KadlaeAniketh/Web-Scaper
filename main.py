import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

# ----------------------------- 
# Local modules (make sure these exist)
# ----------------------------- 
try:
    from scrape import (
        scrape_website,
        extract_body_content,
        clean_body_content,
        split_dom_content,
    )
    from parse_with_groq import parse_with_groq
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    st.warning("⚠️ Scraping modules not found. PDF Q&A only.")

# ----------------------------- 
# LangChain imports
# ----------------------------- 
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------- 
# Environment
# ----------------------------- 
load_dotenv()

# Check for API key in environment or Streamlit secrets
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------------- 
# Embeddings (LOCAL – no token needed)
# ----------------------------- 
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ----------------------------- 
# Streamlit UI Configuration
# ----------------------------- 
st.set_page_config(
    page_title="AI Web Scraper & PDF Q&A", 
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ AI Web Scraper & PDF Q&A")
st.write("Scrape dynamic websites or ask questions from uploaded PDFs using Groq LLMs.")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
models = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

selected_model = st.sidebar.selectbox(
    "🧠 Select Groq Model",
    models,
    index=0
)

# Check API key
if not GROQ_API_KEY:
    st.error("❌ **GROQ_API_KEY is missing!**")
    st.info("""
    **For local development:** Add to `.env` file:
