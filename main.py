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
```
    GROQ_API_KEY=your_api_key_here
```
    
    **For Streamlit Cloud:** Add to app secrets:
    1. Go to app settings
    2. Click "Secrets"
    3. Add: `GROQ_API_KEY = "your_api_key_here"`
    """)
    st.stop()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=selected_model,
    temperature=0.2
)

# ----------------------------- 
# Choose functionality
# ----------------------------- 
if SCRAPING_AVAILABLE:
    choice = st.radio(
        "Choose an action:",
        ["Scrape Website", "Upload PDF & Ask Questions"],
        horizontal=True
    )
else:
    choice = "Upload PDF & Ask Questions"
    st.info("📄 Only PDF Q&A available (scraping modules not found)")

# ===================================================== 
# 🔹 SCRAPE WEBSITE
# ===================================================== 
if choice == "Scrape Website" and SCRAPING_AVAILABLE:
    st.header("🌐 Web Scraper")
    
    url = st.text_input("Enter website URL", placeholder="https://example.com")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        scrape_btn = st.button("🚀 Scrape", type="primary")
    
    if scrape_btn and url:
        if not url.startswith(('http://', 'https://')):
            st.error("❌ Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("🔄 Scraping website..."):
                try:
                    dom_content = scrape_website(url)
                    body_content = extract_body_content(dom_content)
                    cleaned = clean_body_content(body_content)
                    
                    st.session_state.cleaned_content = cleaned
                    st.session_state.scraped_url = url
                    
                    st.success(f"✅ Successfully scraped {len(cleaned)} characters")
                    
                    with st.expander("📄 View Scraped Content"):
                        st.text_area(
                            "Raw HTML Content", 
                            cleaned, 
                            height=400,
                            disabled=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Scraping failed: {str(e)}")
                    st.info("💡 Try checking if the website allows scraping or if the URL is correct.")
    
    # Parse scraped content
    if "cleaned_content" in st.session_state:
        st.divider()
        st.subheader("🔍 Parse Scraped Content")
        
        parse_desc = st.text_area(
            "What information do you want to extract?",
            placeholder="e.g., Extract all product names and prices, or Get all article headlines",
            height=100
        )
        
        if st.button("🎯 Parse Content", type="primary"):
            if not parse_desc.strip():
                st.warning("⚠️ Please describe what you want to extract")
            else:
                with st.spinner("🤖 Parsing with AI..."):
                    try:
                        chunks = split_dom_content(st.session_state.cleaned_content)
                        
                        with st.expander("📦 View Content Chunks"):
                            st.write(f"Total chunks: {len(chunks)}")
                            st.code(chunks[0][:500] + "..." if len(chunks[0]) > 500 else chunks[0])
                        
                        result = parse_with_groq(chunks, parse_desc)
                        
                        if result and result.strip():
                            st.success("✅ Extraction Complete")
                            st.markdown("### 📊 Extracted Information")
                            st.markdown(result)
                            
                            # Download option
                            st.download_button(
                                label="📥 Download Result",
                                data=result,
                                file_name="parsed_content.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("⚠️ No results found. Try refining your description.")
                            
                    except Exception as e:
                        st.error(f"❌ Parsing error: {str(e)}")

# ===================================================== 
# 🔹 PDF RAG Q&A
# ===================================================== 
elif choice == "Upload PDF & Ask Questions":
    st.header("📚 PDF Question & Answer (RAG)")
    
    # Session management
    session_id = st.sidebar.text_input(
        "🆔 Session ID", 
        value="default_session",
        help="Use different session IDs to maintain separate conversation histories"
    )
    
    # Initialize session store
    if "store" not in st.session_state:
        st.session_state.store = {}
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    def get_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        if session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
        st.session_state.chat_messages = []
        st.success("Chat history cleared!")
        st.rerun()
    
    # File upload
    uploaded_files = st.file_uploader(
        "📄 Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to ask questions about"
    )
    
    if uploaded_files:
        with st.spinner("📖 Processing PDFs..."):
            try:
                docs = []
                
                # Use temporary directory for file handling
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        # Save uploaded file to temp directory
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.read())
                        
                        # Load PDF
                        loader = PyPDFLoader(temp_path)
                        docs.extend(loader.load())
                    
                    st.success(f"✅ Loaded {len(docs)} pages from {len(uploaded_files)} PDF(s)")
                    
                    # Split documents
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    splits = splitter.split_documents(docs)
                    
                    st.info(f"📦 Created {len(splits)} text chunks for processing")
                    
                    # Create vector store
                    with st.spinner("🧠 Building knowledge base..."):
                        vectorstore = FAISS.from_documents(
                            documents=splits,
                            embedding=embeddings
                        )
                        retriever = vectorstore.as_retriever(
                            search_kwargs={"k": 4}
                        )
                    
                    # ----------------------------- 
                    # History-aware retriever
                    # ----------------------------- 
                    contextualize_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Given a chat history and the latest user question, "
                                   "formulate a standalone question which can be understood "
                                   "without the chat history. Do NOT answer the question, "
                                   "just reformulate it if needed and otherwise return it as is."),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}")
                    ])
                    
                    history_aware_retriever = create_history_aware_retriever(
                        llm, retriever, contextualize_prompt
                    )
                    
                    # ----------------------------- 
                    # QA chain
                    # ----------------------------- 
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an assistant for question-answering tasks. "
                                   "Use the following pieces of retrieved context to answer "
                                   "the question. If you don't know the answer, say that you "
                                   "don't know. Use three sentences maximum and keep the "
                                   "answer concise.\n\nContext:\n{context}"),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}")
                    ])
                    
                    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
                    
                    rag_chain = create_retrieval_chain(
                        history_aware_retriever,
                        qa_chain
                    )
                    
                    conversational_rag_chain = RunnableWithMessageHistory(
                        rag_chain,
                        get_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer"
                    )
                    
                    st.success("✅ Ready to answer questions!")
                    
                    # ----------------------------- 
                    # Display chat history
                    # ----------------------------- 
                    st.divider()
                    st.subheader("💬 Chat")
                    
                    # Display previous messages
                    for msg in st.session_state.chat_messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    
                    # Chat input
                    user_question = st.chat_input("Ask a question about the PDFs...")
                    
                    if user_question:
                        # Add user message to chat
                        st.session_state.chat_messages.append({
                            "role": "user",
                            "content": user_question
                        })
                        
                        with st.chat_message("user"):
                            st.markdown(user_question)
                        
                        # Get AI response
                        with st.chat_message("assistant"):
                            with st.spinner("🤔 Thinking..."):
                                try:
                                    response = conversational_rag_chain.invoke(
                                        {"input": user_question},
                                        config={
                                            "configurable": {"session_id": session_id}
                                        }
                                    )
                                    
                                    answer = response["answer"]
                                    st.markdown(answer)
                                    
                                    # Add assistant message to chat
                                    st.session_state.chat_messages.append({
                                        "role": "assistant",
                                        "content": answer
                                    })
                                    
                                    # Show source documents in expander
                                    with st.expander("📚 View source excerpts"):
                                        for i, doc in enumerate(response.get("context", []), 1):
                                            st.markdown(f"**Source {i}:**")
                                            st.text(doc.page_content[:300] + "...")
                                            st.divider()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ PDF processing error: {str(e)}")
                st.info("💡 Make sure your PDFs are valid and not corrupted.")

# ----------------------------- 
# Footer
# ----------------------------- 
st.divider()
st.caption("Built with Streamlit, LangChain, and Groq 🚀")
