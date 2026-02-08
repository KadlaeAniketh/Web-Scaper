import streamlit as st
import os

from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
)
from parse import parse_with_ollama

# ---------------- LANGCHAIN (STABLE ONLY) ----------------
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ---------------- API KEYS (STREAMLIT CLOUD SAFE) ----------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["HF_TOKEN"] = st.secrets.get("HF_TOKEN", "")

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Web Scraper & RAG", layout="wide")
st.title("AI-Powered Web Scraper & RAG Q&A App")
st.write("Choose to scrape a website or upload PDFs for Q&A.")

choice = st.radio("Choose an action:", ["Scrape Website", "Upload PDF"])

# =========================================================
# 🔹 SCRAPE WEBSITE
# =========================================================
if choice == "Scrape Website":
    url = st.text_input("Enter the website URL")

    if st.button("Scrape Website"):
        try:
            dom = scrape_website(url)
            body = extract_body_content(dom)
            cleaned = clean_body_content(body)
            st.session_state.cleaned_content = cleaned

            with st.expander("Scraped Content"):
                st.text_area("Cleaned HTML Text", cleaned, height=300)

        except Exception as e:
            st.error(f"❌ Scraping error: {e}")

    if "cleaned_content" in st.session_state:
        parse_desc = st.text_area("Describe what to extract")
        if st.button("Parse Content") and parse_desc:
            chunks = split_dom_content(st.session_state.cleaned_content)
            result = parse_with_ollama(chunks, parse_desc)
            st.success("Extracted Content:")
            st.text(result)

# =========================================================
# 🔹 PDF UPLOAD + RAG
# =========================================================
elif choice == "Upload PDF":

    models = [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama-3.1-70b-specdec",
        "llama-3.1-8b-instant",
        "Gemma2-9b-It",
        "Gemma-7b-It",
    ]

    selected_model = st.sidebar.selectbox("Select Groq Model", models)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=selected_model
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        try:
            docs = []
            for f in uploaded_files:
                temp_path = f"./temp_{f.name}"
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(f.read())

                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )

            retriever = vectorstore.as_retriever()

            # ✅ STABLE RAG CHAIN
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=False
            )

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_q = st.text_input("Ask a question:")
            if user_q:
                response = qa_chain({
                    "question": user_q,
                    "chat_history": st.session_state.chat_history
                })

                st.session_state.chat_history.append(
                    (user_q, response["answer"])
                )

                st.success("Answer:")
                st.write(response["answer"])

        except Exception as e:
            st.error(f"❌ PDF error: {e}")
