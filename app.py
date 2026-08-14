import os
import uuid
import tempfile
import pymupdf
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import streamlit as st
from dotenv import load_dotenv

# --- Environment & Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise Gemini RAG",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ChatGPT / Gemini Dark Modern Theme Styling ---
st.markdown("""
    <style>
    /* Global Base */
    .stApp {
        background-color: #131314;
        color: #e3e3e3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1f20;
        border-right: 1px solid #2d2f31;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        background-color: #2b2c2f;
        color: #f1f3f4;
        border: 1px solid #3c4043;
        transition: all 0.2s ease;
        text-align: left;
    }
    .stButton>button:hover {
        background-color: #37393b;
        border-color: #5f6368;
        color: #ffffff;
    }
    
    /* Primary / New Chat Button */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #1a73e8 !important;
        border: none !important;
        color: white !important;
        font-weight: 500;
    }
    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #1557b0 !important;
    }

    /* Active Tab / Chat Session Item */
    .session-btn {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 4px;
        font-size: 0.88rem;
        cursor: pointer;
    }

    /* Citation Tags */
    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #282a2c;
        color: #8ab4f8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        margin-right: 6px;
        margin-top: 8px;
        border: 1px solid #3c4043;
    }
    
    /* Chat Input Bar */
    div[data-testid="stChatInput"] {
        background-color: transparent;
    }
    div[data-testid="stChatInput"] > div {
        background-color: #1e1f20;
        border: 1px solid #3c4043;
        border-radius: 14px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Services Implementation ---
class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_with_metadata(self, file_path: str, filename: str):
        doc = pymupdf.open(file_path)
        documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                continue
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1
                    }
                })
        doc.close()
        return documents


class VectorStoreManager:
    def __init__(self, collection_name: str = "rag_documents"):
        os.makedirs("./vector_db", exist_ok=True)
        self.client = chromadb.PersistentClient(path="./vector_db")
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )

    def add_documents(self, docs):
        texts = [doc["text"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]
        ids = [f"{doc['metadata']['source']}_p{doc['metadata']['page']}_{i}" for i, doc in enumerate(docs)]
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, top_k: int = 3):
        return self.collection.query(query_texts=[query], n_results=top_k)


class RAGEngine:
    def __init__(self, vector_store: VectorStoreManager, api_key: str):
        self.vector_store = vector_store
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = 1200
        self.client = genai.Client(api_key=api_key)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate_response(self, query: str):
        try:
            # 1. Similarity Retrieval
            search_results = self.vector_store.similarity_search(query, top_k=3)
            retrieved_texts = search_results.get('documents', [[]])[0]
            retrieved_meta = search_results.get('metadatas', [[]])[0]
            distances = search_results.get('distances', [[]])[0]

            # 2. Threshold early exit for out-of-scope queries
            SIMILARITY_THRESHOLD = 1.25
            if not distances or distances[0] > SIMILARITY_THRESHOLD:
                return {
                    "answer": "Mujhe provided documents mein is question ka relevant answer nahi mila.",
                    "sources": []
                }

            # 3. Dynamic context budgeting
            current_tokens = 0
            context_blocks = []
            sources = []

            for text, meta in zip(retrieved_texts, retrieved_meta):
                text_tokens = self.count_tokens(text)
                if current_tokens + text_tokens > self.max_context_tokens:
                    break
                context_blocks.append(f"[{meta['source']} P.{meta['page']}]: {text}")
                sources.append(f"{meta['source']} - Page {meta['page']}")
                current_tokens += text_tokens

            formatted_context = "\n\n".join(context_blocks)

            # 4. Strict Grounding Prompt
            system_prompt = f"""You are an enterprise AI assistant.
Answer accurately using ONLY the context provided below.
If the information is not in the context, strictly state: "Mujhe provided documents mein is question ka relevant answer nahi mila."

Context:
{formatted_context}

User Query: {query}
Answer:"""

            response = self.client.models.generate_content(
                model='gemini-3.5-flash',
                contents=system_prompt,
                config={'max_output_tokens': 1000, 'temperature': 0.2}
            )

            answer_text = response.text.strip() if response.text else ""
            fallback_msg = "Mujhe provided documents mein is question ka relevant answer nahi mila."

            if fallback_msg in answer_text or not answer_text:
                return {"answer": fallback_msg, "sources": []}

            return {
                "answer": answer_text,
                "sources": list(set(sources))
            }
        except Exception as e:
            return {"answer": f"System Error: {str(e)}", "sources": []}


# --- Singleton Resources ---
@st.cache_resource
def init_system():
    doc_processor = DocumentProcessor()
    vector_store = VectorStoreManager()
    rag_engine = RAGEngine(vector_store, GEMINI_API_KEY)
    return doc_processor, vector_store, rag_engine

doc_processor, vector_store, rag_engine = init_system()


# --- Session State Management (Chat Tabs / Sessions) ---
if "sessions" not in st.session_state:
    initial_id = str(uuid.uuid4())
    st.session_state.sessions = {
        initial_id: {
            "title": "New Chat",
            "messages": []
        }
    }
    st.session_state.active_session = initial_id

def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {
        "title": "New Chat",
        "messages": []
    }
    st.session_state.active_session = new_id

# --- Sidebar: Tabs, History & Ingestion ---
with st.sidebar:
    st.markdown("### ✨ Shikra RAG Studio")
    
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.markdown("---")
    
    # 2. Chat Sessions History (Tabs)
    st.markdown("<div style='color: #9aa0a6; font-size: 0.85rem; margin-bottom: 8px;'>Recent Chats</div>", unsafe_allow_html=True)
    
    session_ids = list(st.session_state.sessions.keys())
    for s_id in reversed(session_ids):
        session_data = st.session_state.sessions[s_id]
        is_active = (s_id == st.session_state.active_session)
        
        # Display session tab button with dynamic prefix
        btn_label = f"💬 {session_data['title'][:20]}" if not is_active else f"👉 {session_data['title'][:20]}"
        
        col_tab, col_del = st.columns([0.82, 0.18])
        with col_tab:
            if st.button(btn_label, key=f"tab_{s_id}", use_container_width=True):
                st.session_state.active_session = s_id
                st.rerun()
        with col_del:
            if st.button("✕", key=f"del_{s_id}", help="Delete chat", use_container_width=True):
                if len(st.session_state.sessions) > 1:
                    del st.session_state.sessions[s_id]
                    if st.session_state.active_session == s_id:
                        st.session_state.active_session = list(st.session_state.sessions.keys())[0]
                    st.rerun()

    st.markdown("---")

    # 3. Document Ingestion Module
    with st.expander("📁 Document Vault", expanded=False):
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
        if st.button("⚡ Ingest Document", use_container_width=True):
            if uploaded_file and GEMINI_API_KEY:
                with st.spinner("Processing & Indexing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        docs = doc_processor.extract_text_with_metadata(tmp_path, uploaded_file.name)
                        vector_store.add_documents(docs)
                        st.success(f"Indexed {len(docs)} chunks successfully!")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
            elif not uploaded_file:
                st.warning("Select a PDF file first.")


# --- Main Chat Area ---
active_id = st.session_state.active_session
current_session = st.session_state.sessions[active_id]

# Header
st.markdown(f"<h3 style='margin-bottom: 0px;'>{current_session['title']}</h3>", unsafe_allow_html=True)
st.caption("Powered by ChromaDB Vector Search & Gemini 3.5 Flash Grounding")
st.markdown("---")

# Render active session messages
for msg in current_session["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            sources_html = "".join([f'<span class="source-tag">📄 {src}</span>' for src in msg["sources"]])
            st.markdown(sources_html, unsafe_allow_html=True)

# User Query Interaction
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Update session title from first user query
    if len(current_session["messages"]) == 0:
        current_session["title"] = prompt[:24] + ("..." if len(prompt) > 24 else "")

    # 2. Append User Message
    current_session["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_engine.generate_response(prompt)
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            if sources:
                sources_html = "".join([f'<span class="source-tag">📄 {src}</span>' for src in sources])
                st.markdown(sources_html, unsafe_allow_html=True)

            current_session["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })