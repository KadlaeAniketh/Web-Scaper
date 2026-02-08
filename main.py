import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Local modules
# -----------------------------
try:
    from scrape import scrape_website, extract_body_content, clean_body_content, split_dom_content
    from parse_with_groq import parse_with_groq
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    st.warning("⚠️ Scraping modules not found. PDF Q&A only.")

# -----------------------------
# LangChain / Groq / PDF
# -----------------------------
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
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
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# -----------------------------
# Embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Web Scraper & PDF Q&A", page_icon="🕸️", layout="wide")
st.title("🕸️ AI Web Scraper & PDF Q&A")
st.write("Scrape websites or ask questions from uploaded PDFs using Groq LLMs.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
models = ["llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]
selected_model = st.sidebar.selectbox("🧠 Select Groq Model", models, index=0)

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY missing! Add it to Streamlit secrets or .env")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=selected_model, temperature=0.2)

# Choose functionality
if SCRAPING_AVAILABLE:
    choice = st.radio("Choose an action:", ["Scrape Website", "Upload PDF & Ask Questions"], horizontal=True)
else:
    choice = "Upload PDF & Ask Questions"
    st.info("📄 Only PDF Q&A available (scraping modules not found)")

# ===========================
# Web Scraper
# ===========================
if choice == "Scrape Website" and SCRAPING_AVAILABLE:
    st.header("🌐 Web Scraper")
    url = st.text_input("Enter website URL", placeholder="https://example.com")
    
    if st.button("🚀 Scrape") and url:
        if not url.startswith(("http://", "https://")):
            st.error("❌ URL must start with http:// or https://")
        else:
            try:
                st.spinner("🔄 Scraping website...")
                dom_content = scrape_website(url)
                body_content = extract_body_content(dom_content)
                cleaned = clean_body_content(body_content)
                
                st.session_state.cleaned_content = cleaned
                st.session_state.scraped_url = url
                
                st.success(f"✅ Scraped {len(cleaned)} characters")
                with st.expander("📄 View Scraped Content"):
                    st.text_area("Raw HTML", cleaned, height=400, disabled=True)
            except Exception as e:
                st.error(f"❌ Scraping failed: {str(e)}")

    # Parse scraped content
    if "cleaned_content" in st.session_state:
        st.subheader("🔍 Parse Scraped Content")
        parse_desc = st.text_area("Describe what to extract", placeholder="Extract all titles or job listings")
        
        if st.button("🎯 Parse Content") and parse_desc.strip():
            chunks = split_dom_content(st.session_state.cleaned_content)
            result = parse_with_groq(chunks, parse_desc)
            
            if result:
                st.success("✅ Extraction complete")
                st.markdown(result)
                st.download_button("📥 Download Result", result, "parsed_content.txt", "text/plain")
            else:
                st.warning("⚠️ No results found. Try refining the description.")

# ===========================
# PDF RAG Q&A
# ===========================
elif choice == "Upload PDF & Ask Questions":
    st.header("📚 PDF Q&A (RAG)")
    session_id = st.sidebar.text_input("🆔 Session ID", "default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    def get_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    if st.sidebar.button("🗑️ Clear Chat History"):
        if session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
        st.session_state.chat_messages = []
        st.success("Chat history cleared!")
        st.experimental_rerun()

    uploaded_files = st.file_uploader("📄 Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("📖 Processing PDFs..."):
            docs = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for f in uploaded_files:
                    temp_path = os.path.join(temp_dir, f.name)
                    with open(temp_path, "wb") as file:
                        file.write(f.read())
                    loader = PyPDFLoader(temp_path)
                    docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)
            
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # History-aware retriever
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "Formulate a standalone question from the chat history without answering."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

            # QA Chain
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer concisely using the context:\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            st.success("✅ Ready to answer questions!")

            # Display chat history
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_question = st.chat_input("Ask a question about the PDFs...")
            if user_question:
                st.session_state.chat_messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        response = conversational_rag_chain.invoke({"input": user_question}, config={"configurable": {"session_id": session_id}})
                        answer = response["answer"]
                        st.markdown(answer)
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("Built with Streamlit, LangChain, and Groq 🚀")
