import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import re
from typing import List, Generator
from pydantic import Field
import time

load_dotenv()

# Constants
TEMPERATURE = 0.1
LLM_MODEL = 'llama-3.1-8b-instant'
RETRIEVAL_K = 8
FINAL_K = 4
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# Enhanced CSS styling with minimal gaps and light colors
def load_css():
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Light color variables */
    :root {
        --primary-color: #10b981;
        --primary-light: #34d399;
        --primary-dark: #059669;
        --secondary-color: #6b7280;
        --accent-color: #06b6d4;
        --success-color: #059669;
        --warning-color: #f59e0b;
        --error-color: #ef4444;

        /* Light theme colors */
        --bg-primary: #fefefe;
        --bg-secondary: #f9fafb;
        --bg-tertiary: #f3f4f6;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
        --shadow-color: rgba(0, 0, 0, 0.05);

        /* Card backgrounds */
        --card-bg: rgba(255, 255, 255, 0.98);
        --card-hover-bg: rgba(249, 250, 251, 0.98);
        --sidebar-bg: rgba(243, 244, 246, 0.95);

        /* Light gradient background */
        --gradient-start: #ecfdf5;
        --gradient-end: #d1fae5;
    }

    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #111827;
            --bg-secondary: #1f2937;
            --bg-tertiary: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --border-color: #374151;
            --shadow-color: rgba(0, 0, 0, 0.2);

            --card-bg: rgba(31, 41, 55, 0.95);
            --card-hover-bg: rgba(55, 65, 81, 0.98);
            --sidebar-bg: rgba(17, 24, 39, 0.95);

            --gradient-start: #064e3b;
            --gradient-end: #065f46;
        }
    }

    .stApp[data-theme="dark"] {
        --bg-primary: #111827;
        --bg-secondary: #1f2937;
        --bg-tertiary: #374151;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --border-color: #374151;
        --shadow-color: rgba(0, 0, 0, 0.2);

        --card-bg: rgba(31, 41, 55, 0.95);
        --card-hover-bg: rgba(55, 65, 81, 0.98);
        --sidebar-bg: rgba(17, 24, 39, 0.95);

        --gradient-start: #064e3b;
        --gradient-end: #065f46;
    }

    /* Global styling with light gradient background */
    .stApp {
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* MINIMIZE TOP GAP - Main container styling */
    .main .block-container {
        padding-top: 1rem !important;  /* Reduced from 2rem */
        padding-bottom: 1rem !important;
        max-width: 1200px;
        background: var(--card-bg);
        border-radius: 16px;
        box-shadow: 0 10px 30px var(--shadow-color);
        backdrop-filter: blur(10px);
        margin: 0.5rem auto !important;  /* Reduced margin */
        border: 1px solid var(--border-color);
    }

    /* Reduce header margins */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;  /* Slightly smaller */
        font-weight: 700;
        margin-bottom: 0.25rem !important;  /* Reduced margin */
        margin-top: 0 !important;
        animation: fadeInDown 0.8s ease-out;
    }

    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem !important;  /* Reduced margin */
        margin-top: 0 !important;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }

    /* Reduce section spacing */
    .stMarkdown h3 {
        margin-top: 1rem !important;  /* Reduced from default */
        margin-bottom: 0.75rem !important;
    }

    /* Animation keyframes */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    /* File uploader styling with light theme */
    .stFileUploader > div {
        background: var(--bg-tertiary) !important;
        border: 2px dashed var(--primary-light) !important;
        border-radius: 12px;
        padding: 1.5rem !important;  /* Slightly reduced padding */
        text-align: center;
        transition: all 0.3s ease;
        color: var(--text-primary) !important;
    }

    .stFileUploader > div:hover {
        border-color: var(--primary-color) !important;
        background: var(--card-hover-bg) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.12);
    }

    /* Button styling with green theme */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 3px 12px rgba(16, 185, 129, 0.25) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35) !important;
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary-color) 100%) !important;
    }

    /* Sidebar styling */
    .css-1d391kg, .css-1outpf7, .stSidebar {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    .sidebar-content {
        padding: 1rem;
        background: var(--card-bg);
        border-radius: 10px;
        margin: 0.4rem 0;
        box-shadow: 0 2px 8px var(--shadow-color);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    /* Feature cards with reduced spacing */
    .feature-card {
        background: var(--card-bg) !important;
        padding: 1.25rem !important;  /* Slightly reduced */
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 0.4rem 0 !important;  /* Reduced margin */
        transition: all 0.3s ease;
        color: var(--text-primary);
        box-shadow: 0 3px 12px var(--shadow-color);
    }

    .feature-card:hover {
        border-color: var(--primary-light);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.12);
        background: var(--card-hover-bg) !important;
        transform: translateY(-3px);
    }

    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
        display: block;
    }

    /* Metric cards with light theme */
    .metric-card {
        background: var(--card-bg) !important;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 3px 12px var(--shadow-color);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 20px var(--shadow-color);
        border-color: var(--primary-light);
        background: var(--card-hover-bg) !important;
    }

    .metric-card h2 {
        color: var(--text-primary) !important;
        margin: 0.2rem 0 !important;
    }

    .metric-card h3 {
        margin: 0 !important;
    }

    .metric-card p {
        color: var(--text-secondary) !important;
        margin: 0 !important;
        font-size: 0.9rem;
    }

    /* Chat message styling */
    .stChatMessage {
        background: var(--card-bg) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px var(--shadow-color) !important;
        margin: 0.8rem 0 !important;  /* Reduced margin */
        animation: slideIn 0.5s ease-out;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-15px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Success/Info messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--success-color) 0%, #047857 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(5, 150, 105, 0.25) !important;
    }

    .stInfo {
        background: linear-gradient(135deg, var(--accent-color) 0%, #0891b2 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }

    .streamlit-expanderContent {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid var(--border-color) !important;
        padding: 0.7rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
        background: var(--card-bg) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
        border-radius: 8px !important;
    }

    /* Custom status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    .status-connected { background: var(--success-color); }
    .status-processing { background: var(--warning-color); }
    .status-ready { background: var(--accent-color); }

    /* Footer styling */
    .footer-card {
        background: var(--card-bg) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-color);
        box-shadow: 0 3px 12px var(--shadow-color);
    }

    /* Text color overrides */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }

    .stMarkdown small {
        color: var(--text-secondary) !important;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header { font-size: 2rem !important; }
        .sub-header { font-size: 1rem !important; }
        .main .block-container { 
            margin: 0.25rem auto !important; 
            padding: 0.75rem !important; 
        }
    }

    /* Additional spacing reductions */
    .element-container {
        margin-bottom: 0.75rem !important;
    }

    .stColumns {
        gap: 0.75rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced header component with reduced spacing
def render_header():
    st.markdown("""
    <div class="main-header">
        🧠 Enhanced RAG Assistant
    </div>
    <div class="sub-header">
        🚀 Smart Document Q&A with Hybrid Search & AI-Powered Streaming
    </div>
    """, unsafe_allow_html=True)

# Status indicator component
def status_indicator(status_type, text):
    if status_type == "connected":
        return f'<span class="status-indicator status-connected"></span>{text}'
    elif status_type == "processing":
        return f'<span class="status-indicator status-processing"></span>{text}'
    elif status_type == "ready":
        return f'<span class="status-indicator status-ready"></span>{text}'

# Feature showcase component
def render_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🔍</span>
            <strong>Hybrid Search</strong><br>
            <small>Combines semantic & keyword search with intelligent reranking</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <strong>Real-time Streaming</strong><br>
            <small>Watch responses appear in real-time as AI processes your query</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <strong>Context Awareness</strong><br>
            <small>Maintains conversation history for coherent multi-turn dialogues</small>
        </div>
        """, unsafe_allow_html=True)

# Enhanced metrics display
def render_metrics(doc_count=0, process_time=0, chunks_retrieved=0):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color: var(--primary-color);">📄</h3>
            <h2 style="margin:0;">{doc_count}</h2>
            <p style="margin:0;">Documents</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color: var(--success-color);">⏱️</h3>
            <h2 style="margin:0;">{process_time:.1f}s</h2>
            <p style="margin:0;">Response Time</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color: var(--warning-color);">🎯</h3>
            <h2 style="margin:0;">{chunks_retrieved}</h2>
            <p style="margin:0;">Chunks Retrieved</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color: var(--accent-color);">🔄</h3>
            <h2 style="margin:0;">{len(st.session_state.get('chat_history', []))//2}</h2>
            <p style="margin:0;">Conversations</p>
        </div>
        """, unsafe_allow_html=True)

# Load CSS and configure page
load_css()
st.set_page_config(
    page_title="Enhanced RAG Assistant", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

render_header()

# API Configuration
try:
    load_dotenv()  # For local development
    api_key = os.getenv("GROQ_API_KEY")
except:
    # For Streamlit Cloud deployment
    api_key = st.secrets["GROQ_API_KEY"]

if api_key:
    llm = ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=TEMPERATURE)
    st.sidebar.markdown(f"""
    <div class="sidebar-content">
        {status_indicator("connected", "API Connected Successfully")}
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.error("❌ GROQ API Key not found")

# Feature showcase with reduced spacing
st.markdown("### ✨ **Key Features**")
render_features()

# File upload section
st.markdown("### 📚 **Document Upload**")
uploaded_files = st.file_uploader(
    "Drop your PDF documents here or click to browse",
    type='pdf', 
    accept_multiple_files=True,
    help="Upload one or more PDF documents to create your knowledge base"
)

# [Keep your existing SimpleQueryExpander and EnhancedHybridRetriever classes unchanged]

class SimpleQueryExpander:
    """Always on query expansion using LLM"""
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def expand_query(self, query: str) -> List[str]:
        expansion_prompt = PromptTemplate.from_template("""
Create 1 alternative phrasing of this query to improve document search:

Original: "{query}"
Generate 1 variation using different keywords or question structure.
Alternative: """)
        try:
            chain = expansion_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"query": query})
            alternative = response.strip().replace('"', '').replace("Alternative:", "").strip()
            if alternative and len(alternative) > 10 and alternative != query:
                return [query, alternative]
        except:
            pass
        return [query]

class EnhancedHybridRetriever(BaseRetriever):
    """Permanent Hybrid + rerank retriever"""
    vectorstore: any = Field(description="FAISS vectorstore")
    bm25_retriever: any = Field(description="BM25 retriever")
    llm_instance: any = Field(description="LLM instance")
    retrieval_k: int = Field(default=RETRIEVAL_K, description="Number of docs to retrieve")
    final_k: int = Field(default=FINAL_K, description="Final number of docs")
    query_expander: any = Field(default=None, description="Query expander")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_expander = SimpleQueryExpander(self.llm_instance)

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        queries = self.query_expander.expand_query(query)
        all_docs = []

        for q in queries:
            bm25_docs = self.bm25_retriever.get_relevant_documents(q)[:self.retrieval_k//2]
            vector_docs = self.vectorstore.as_retriever(
                search_kwargs={"k": self.retrieval_k//2}
            ).get_relevant_documents(q)
            all_docs.extend(bm25_docs + vector_docs)

        unique_docs = self._remove_duplicates(all_docs)
        ranked_docs = self._rerank(query, unique_docs[:min(8, len(unique_docs))])
        return ranked_docs[:self.final_k]

    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in docs:
            content_hash = hash(doc.page_content[:100].strip())
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)
        return unique

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if len(docs) <= self.final_k:
            return docs

        doc_texts = [f"{i+1}. {doc.page_content[:200]}..." for i, doc in enumerate(docs)]
        docs_text = "\n\n".join(doc_texts)

        prompt = PromptTemplate.from_template("""
Select the {final_k} most relevant document chunks for this query:

Query: {query}

Documents:
{documents}

Return only the numbers of the most relevant documents, comma-separated (e.g. 1,3,5,2):
""")

        try:
            chain = prompt | self.llm_instance | StrOutputParser()
            response = chain.invoke({"query": query, "documents": docs_text, "final_k": self.final_k})

            indices = [int(x)-1 for x in re.findall(r'\d+', response) if 0 <= int(x)-1 < len(docs)]
            ranked_docs = [docs[i] for i in indices[:self.final_k]]

            while len(ranked_docs) < self.final_k and len(ranked_docs) < len(docs):
                for doc in docs:
                    if doc not in ranked_docs:
                        ranked_docs.append(doc)
                        if len(ranked_docs) == self.final_k:
                            break

            return ranked_docs
        except:
            return docs[:self.final_k]

# Main processing logic
if uploaded_files and api_key:
    documents = []

    # Enhanced progress display
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("🔄 Processing documents..."):
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.markdown(f"{status_indicator('processing', f'Processing {uploaded_file.name}...')}", unsafe_allow_html=True)
            progress_bar.progress((i + 1) / len(uploaded_files))

            temp_path = f"./temp_{uploaded_file.name}.pdf"
            with open(temp_path, 'wb') as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
            os.remove(temp_path)

    progress_bar.empty()
    status_text.empty()

    st.success(f"✅ Successfully processed {len(uploaded_files)} documents with {len(documents)} pages!")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "rag_chain" not in st.session_state:
        with st.spinner("🔧 Building advanced RAG system..."):
            # Create embeddings and vector store
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP, 
                length_function=len
            )
            chunked_docs = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(chunked_docs, embedding)

            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(chunked_docs)
            bm25_retriever.k = RETRIEVAL_K // 2

            # Create enhanced hybrid retriever
            enhanced_retriever = EnhancedHybridRetriever(
                vectorstore=vectorstore,
                bm25_retriever=bm25_retriever,
                llm_instance=llm,
                retrieval_k=RETRIEVAL_K,
                final_k=FINAL_K
            )

            # Create history-aware retriever
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given chat history and user question, create a standalone question. If already standalone, return as is."),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm, enhanced_retriever, contextualize_prompt
            )

            # Create answer generation chain
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are a helpful assistant that answers questions using the provided context. "
                 "Use only information from the context. If the context doesn't contain enough "
                 "information, say so clearly. Provide detailed, well-structured answers."),
                MessagesPlaceholder("chat_history"),
                ("user", "Context: {context}\n\nQuestion: {input}\n\nAnswer:")
            ])

            qa_chain = create_stuff_documents_chain(llm, answer_prompt)
            st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        st.success("🎯 Advanced RAG system ready with Hybrid Search & Streaming!")

    # Display metrics
    st.markdown("### 📊 **System Metrics**")
    render_metrics(
        doc_count=len(documents),
        process_time=0,
        chunks_retrieved=FINAL_K
    )

    # Enhanced Chat Interface
    st.markdown("### 💬 **Intelligent Chat Interface**")

    # Display chat history
    for i in range(0, len(st.session_state.chat_history), 2):
        if i+1 < len(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(st.session_state.chat_history[i][1])

            with st.chat_message("assistant"):
                st.write(st.session_state.chat_history[i+1][1])

    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            chain_input = {
                "input": user_query,
                "chat_history": st.session_state.chat_history
            }

            start_time = time.time()

            try:
                def generate_response():
                    for chunk in st.session_state.rag_chain.stream(chain_input):
                        if 'answer' in chunk:
                            yield chunk['answer']

                full_response = st.write_stream(generate_response())
                process_time = time.time() - start_time

                # Enhanced response metrics
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 0.6rem; background: var(--card-bg); border-radius: 10px; border-left: 4px solid var(--primary-color); box-shadow: 0 2px 6px var(--shadow-color); border: 1px solid var(--border-color);">
                    <span style="color: var(--text-primary); font-size: 0.9rem;"> ⏱️ <strong>{process_time:.1f}s</strong> | 🎯 <strong>{FINAL_K} chunks</strong> | ⚡ <strong>Streaming enabled</strong></span>
                </div>
                """, unsafe_allow_html=True)

                # Update chat history
                st.session_state.chat_history.extend([
                    ("user", user_query),
                    ("assistant", full_response)
                ])

                # Update metrics
                render_metrics(
                    doc_count=len(documents),
                    process_time=process_time,
                    chunks_retrieved=FINAL_K
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3 style="color: var(--text-primary); margin: 0;">🎛️ System Controls</h3>
    </div>
    """, unsafe_allow_html=True)

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        if "chat_history" in st.session_state:
            st.session_state.chat_history = []
            st.rerun()

    # Enhanced system information
    with st.expander("⚙️ System Configuration", expanded=True):
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--text-primary);">
        <strong>🤖 Model:</strong> {LLM_MODEL}<br>
        <strong>🌡️ Temperature:</strong> {TEMPERATURE}<br>
        <strong>🔗 Embeddings:</strong> all-MiniLM-L6-v2<br>
        <strong>📄 Chunk Size:</strong> {CHUNK_SIZE}<br>
        <strong>🔄 Overlap:</strong> {CHUNK_OVERLAP}<br>
        <strong>🎯 Retrieval:</strong> {RETRIEVAL_K} → {FINAL_K}<br>
        <strong>🔍 Hybrid Search:</strong> ✅<br>
        <strong>🏆 Reranking:</strong> ✅<br>
        <strong>⚡ Streaming:</strong> ✅
        </div>
        """, unsafe_allow_html=True)

    # Chat history
    if st.session_state.get('chat_history'):
        with st.expander("💬 Chat History", expanded=False):
            for i in range(0, len(st.session_state.chat_history), 2):
                if i+1 < len(st.session_state.chat_history):
                    st.markdown(f"""
                    <div style="margin: 0.4rem 0; padding: 0.6rem; background: var(--card-bg); border-radius: 6px; border-left: 3px solid var(--primary-color); border: 1px solid var(--border-color); box-shadow: 0 1px 4px var(--shadow-color);">
                        <strong style="color: var(--text-primary); font-size: 0.85rem;">Q:</strong> <span style="color: var(--text-secondary); font-size: 0.85rem;">{st.session_state.chat_history[i][1][:50]}...</span>
                    </div>
                    """, unsafe_allow_html=True)

    # Performance tips
    with st.expander("💡 Performance Tips"):
        st.markdown("""
        <div style="color: var(--text-primary); font-size: 0.9rem; line-height: 1.4;">
        • <strong>Upload Related Documents:</strong> Better context leads to better answers<br><br>
        • <strong>Ask Specific Questions:</strong> Detailed queries get detailed responses<br><br>
        • <strong>Use Follow-ups:</strong> The system remembers conversation context<br><br>
        • <strong>Multiple Documents:</strong> Upload various sources for comprehensive answers
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-content" style="text-align: center; margin-top: 1.5rem;">
        <p style="color: var(--primary-color); font-weight: 600; margin: 0;">🎯 Optimized for Accuracy</p>
        <p style="font-size: 0.82rem; color: var(--text-secondary); margin: 0.3rem 0 0 0;">
        Advanced hybrid search with intelligent reranking and real-time streaming responses.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div class="footer-card" style="text-align: center; margin-top: 2rem; padding: 1.5rem; border-radius: 12px;">
    <p style="margin: 0; font-size: 0.9rem;">
        🚀 Powered by Advanced RAG Architecture | Built with ❤️ using Streamlit & LangChain
    </p>
</div>
""", unsafe_allow_html=True)
