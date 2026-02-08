import os
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Local modules
# -----------------------------
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
)
from parse_with_groq import parse_with_groq

# -----------------------------
# LangChain imports (NEW API)
# -----------------------------
from langchain_groq import ChatGroq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# Embeddings (LOCAL – no token)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Web Scraper & PDF Q&A", layout="wide")
st.title("🕸️ AI Web Scraper & PDF Q&A (Groq + RAG)")
st.write("Scrape dynamic websites or ask questions from uploaded PDFs.")

choice = st.radio(
    "Choose an action:",
    ["Scrape Website", "Upload PDF"],
    horizontal=True
)

# =====================================================
# 🔹 SCRAPE WEBSITE
# =====================================================
if choice == "Scrape Website":
    url = st.text_input("🌐 Enter website URL")

    if st.button("Scrape Website"):
        try:
            dom_content = scrape_website(url)
            body_content = extract_body_content(dom_content)
            cleaned = clean_body_content(body_content)

            st.session_state.cleaned_content = cleaned

            with st.expander("📄 Scraped HTML Content"):
                st.text_area("HTML Text", cleaned, height=300)

        except Exception as e:
            st.error(f"❌ Scraping failed: {e}")

    if "cleaned_content" in st.session_state:
        parse_desc = st.text_area(
            "Describe what to extract",
            "Extract job titles, headings, or key points"
        )

        if st.button("Parse Content"):
            try:
                chunks = split_dom_content(st.session_state.cleaned_content)
                st.write("📦 Example DOM Chunk:", chunks[:1])

                result = parse_with_groq(chunks, parse_desc)

                if result.strip():
                    st.success("✅ Extracted Result")
                    st.text(result)
                else:
                    st.warning("⚠️ No results found. Try refining the description.")

            except Exception as e:
                st.error(f"❌ Parsing error: {e}")

# =====================================================
# 🔹 PDF RAG Q&A
# =====================================================
elif choice == "Upload PDF":

    models = [
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama-3.1-70b-specdec",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "gemma-7b-it",
    ]

    selected_model = st.sidebar.selectbox(
        "🧠 Select Groq Model",
        models
    )

    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY missing. Add it to .env or Streamlit secrets.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=selected_model
    )

    session_id = st.text_input("🆔 Session ID", "default")

    if "store" not in st.session_state:
        st.session_state.store = {}

    def get_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    uploaded_files = st.file_uploader(
        "📄 Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        try:
            docs = []

            for f in uploaded_files:
                temp_path = f"./temp_{f.name}"
                with open(temp_path, "wb") as tmp:
                    tmp.write(f.read())

                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=500
            )

            splits = splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(
                splits,
                embedding=embeddings
            )

            retriever = vectorstore.as_retriever()

            # -----------------------------
            # History-aware retriever
            # -----------------------------
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "Rephrase the question based on chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm,
                retriever,
                contextualize_prompt
            )

            # -----------------------------
            # QA chain
            # -----------------------------
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Use the following context to answer the question.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            qa_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(
                history_aware_retriever,
                qa_chain
            )

            rag = RunnableWithMessageHistory(
                rag_chain,
                get_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_q = st.text_input("💬 Ask a question about the PDFs")

            if user_q:
                response = rag.invoke(
                    {"input": user_q},
                    config={"configurable": {"session_id": session_id}}
                )

                st.success("🧠 Answer")
                st.write(response["answer"])

        except Exception as e:
            st.error(f"❌ PDF processing error: {e}")
