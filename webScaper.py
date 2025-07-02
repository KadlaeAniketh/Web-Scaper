import streamlit as st
import os
from dotenv import load_dotenv

from scrape import (
    scrape_website,
    extract_body_content,
    clean_body_content,
    split_dom_content,
)
from parse import parse_with_ollama

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
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

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("AI-Powered Web Scraper & RAG Q&A App")
st.write("Choose to scrape a website or upload PDFs for Q&A.")

choice = st.radio("Choose an action:", ["Scrape Website", "Upload PDF"])

# --- Scrape Website ---
if choice == "Scrape Website":
    url = st.text_input("Enter the website URL")
    if st.button("Scrape Website"):
        st.write("Scraping website...")
        dom_content = scrape_website(url)
        body_content = extract_body_content(dom_content)
        cleaned = clean_body_content(body_content)
        st.session_state.cleaned_content = cleaned

        with st.expander("Scraped Content"):
            st.text_area("Cleaned HTML Text", cleaned, height=300)

    if "cleaned_content" in st.session_state:
        parse_desc = st.text_area("Describe what to extract")
        if st.button("Parse Content") and parse_desc:
            st.write("Extracting information...")
            chunks = split_dom_content(st.session_state.cleaned_content)
            result = parse_with_ollama(chunks, parse_desc)
            st.success("Extracted Content:")
            st.text(result)

# --- Upload PDF and Q&A ---
elif choice == "Upload PDF":
    models = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-70b-specdec",
    "llama-3.1-8b-instant",
    "Gemma2-9b-It",
    "Gemma-7b-It"
    ]

    selected_model = st.sidebar.selectbox("Select Groq Model", models)

    if not GROQ_API_KEY:
        st.error("Set GROQ_API_KEY in .env")
    else:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model)
        session_id = st.text_input("Session ID", "default_session")

        if "store" not in st.session_state:
            st.session_state.store = {}

        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            docs = []
            for f in uploaded_files:
                temp_path = f"./temp_{f.name}"
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(f.read())
                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=splits, embedding=embeddings, persist_directory="./chroma_db"
            )
            retriever = vectorstore.as_retriever()

            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "Rephrase the question based on chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Use the retrieved context to answer questions.\n\n{context}"),
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

            user_q = st.text_input("Ask a question:")
            if user_q:
                response = rag.invoke(
                    {"input": user_q},
                    config={"configurable": {"session_id": session_id}}
                )
                st.success("Answer:")
                st.write(response["answer"])