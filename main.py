import streamlit as st
import os

from dotenv import load_dotenv
from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
)
from parse_with_groq import parse_with_groq

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Local version
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# ✅ Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ No token needed for local embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Streamlit UI
st.title("🕸️ AI Web Scraper & PDF Q&A with Groq")
st.write("Scrape websites or ask questions from uploaded PDFs.")

choice = st.radio("Choose an action:", ["Scrape Website", "Upload PDF"])

# --- Scrape Website
if choice == "Scrape Website":
    url = st.text_input("Enter website URL")
    if st.button("Scrape Website"):
        try:
            dom_content = scrape_website(url)
            body_content = extract_body_content(dom_content)
            cleaned = clean_body_content(body_content)
            st.session_state.cleaned_content = cleaned

            with st.expander("Scraped HTML Content"):
                st.text_area("HTML Text", cleaned, height=300)
        except Exception as e:
            st.error(f"❌ Scraping failed: {e}")

    if "cleaned_content" in st.session_state:
        parse_desc = st.text_area("Describe what to extract", "Extract job titles or headers")
        if st.button("Parse Content"):
            try:
                chunks = split_dom_content(st.session_state.cleaned_content)
                st.write("📦 DOM Chunk Example:", chunks[:1])
                result = parse_with_groq(chunks, parse_desc)
                if result.strip():
                    st.success("✅ Extracted:")
                    st.text(result)
                else:
                    st.warning("⚠️ No results. Try refining the description.")
            except Exception as e:
                st.error(f"❌ Parsing Error: {e}")

# --- PDF Upload
elif choice == "Upload PDF":
    models = [
        "llama3-70b-8192", "llama3-8b-8192",
        "llama-3.1-70b-specdec", "llama-3.1-8b-instant",
        "Gemma2-9b-It", "Gemma-7b-It"
    ]
    selected_model = st.sidebar.selectbox("Select Groq Model", models)

    if not GROQ_API_KEY:
        st.error("❌ Add GROQ_API_KEY to your .env or Streamlit secrets.")
    else:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model)
        session_id = st.text_input("Session ID", "default")

        if "store" not in st.session_state:
            st.session_state.store = {}

        uploaded_files = st.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            try:
                docs = []
                for f in uploaded_files:
                    temp_path = f"./temp_{f.name}"
                    with open(temp_path, "wb") as temp_file:
                        temp_file.write(f.read())
                    loader = PyPDFLoader(temp_path)
                    docs.extend(loader.load())

                splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                splits = splitter.split_documents(docs)

                vectorstore = FAISS.from_documents(splits, embedding=embeddings)
                retriever = vectorstore.as_retriever()

                contextualize_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Rephrase the question based on chat history."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_prompt)

                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Use the context below to answer questions.\n\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                qa_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever_chain, qa_chain)

                def get_history(session: str) -> BaseChatMessageHistory:
                    if session not in st.session_state.store:
                        st.session_state.store[session] = ChatMessageHistory()
                    return st.session_state.store[session]

                rag = RunnableWithMessageHistory(
                    rag_chain,
                    get_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )

                user_q = st.text_input("💬 Ask a question:")
                if user_q:
                    response = rag.invoke(
                        {"input": user_q},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.success("🧠 Answer:")
                    st.write(response["answer"])

            except Exception as e:
                st.error(f"❌ PDF Error: {e}")
