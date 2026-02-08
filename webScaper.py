import streamlit as st
import os

from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
)

from parse_with_groq import parse_with_groq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --------------------------------------------------
# 🔐 Secrets (Streamlit Cloud compatible)
# --------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]
os.environ["HF_TOKEN"] = HF_TOKEN

# --------------------------------------------------
# 🔢 Embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 🖥 Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="AI Web Scraper & RAG", layout="wide")
st.title("AI-Powered Web Scraper & RAG Q&A App")

choice = st.radio(
    "Choose an action:",
    ["Scrape Website", "Upload PDF"]
)

# ==================================================
# 🌐 WEBSITE SCRAPING
# ==================================================
if choice == "Scrape Website":
    url = st.text_input("Enter website URL")

    if st.button("Scrape Website") and url:
        st.info("Scraping website...")
        dom_content = scrape_website(url)
        body = extract_body_content(dom_content)
        cleaned = clean_body_content(body)
        st.session_state.cleaned_content = cleaned

        with st.expander("Scraped Content"):
            st.text_area("Cleaned Text", cleaned, height=300)

    if "cleaned_content" in st.session_state:
        instruction = st.text_area("Describe what to extract")

        if st.button("Parse Content") and instruction:
            st.info("Parsing content with Groq...")
            chunks = split_dom_content(st.session_state.cleaned_content)
            result = parse_with_groq(chunks, instruction)
            st.success("Extracted Result")
            st.write(result)

# ==================================================
# 📄 PDF UPLOAD + RAG Q&A
# ==================================================
else:
    models = [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama-3.1-70b-specdec",
        "llama-3.1-8b-instant",
        "Gemma2-9b-It",
        "Gemma-7b-It",
    ]

    selected_model = st.sidebar.selectbox(
        "Select Groq Model",
        models
    )

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=selected_model
    )

    session_id = st.text_input("Session ID", "default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        documents = []

        for file in uploaded_files:
            temp_path = f"./temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        splits = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        retriever = vectorstore.as_retriever()

        # ----------------------------
        # History-aware retriever
        # ----------------------------
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the question using chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_prompt
        )

        # ----------------------------
        # QA Chain
        # ----------------------------
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using the provided context.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_retriever,
            qa_chain
        )

        # ----------------------------
        # Chat History
        # ----------------------------
        def get_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        user_question = st.text_input("Ask a question")

        if user_question:
            response = conversational_rag.invoke(
                {"input": user_question},
                config={
                    "configurable": {"session_id": session_id}
                },
            )

            st.success("Answer")
            st.write(response["answer"])
