import streamlit as st
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for Enterprise Look ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .source-tag {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-top: 5px;
        border: 1px solid #334155;
    }
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Configuration & Constants ---
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Document Ingestion Panel ---
with st.sidebar:
    st.title("⚙️ Knowledge Base")
    st.caption("Upload company policies, manuals, or PDF reports.")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Select a PDF file to process and add to vector database."
    )
    
    if st.button("🚀 Process & Ingest", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing document (Extracting, Chunking, Indexing)..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Success! Indexed {data.get('chunks_indexed', 0)} chunks.")
                        st.toast("Knowledge Base Updated Successfully!", icon="🎉")
                    else:
                        st.error(f"❌ Ingestion Failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to FastAPI server. Ensure backend is running.")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")
        else:
            st.warning("⚠️ Please select a PDF file first.")
            
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Server Health Check
    try:
        health_check = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/docs")
        if health_check.status_code == 200:
            st.success("● Backend API: Online")
    except:
        st.error("● Backend API: Offline")
        
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Area ---
st.title("🤖 Enterprise RAG Chatbot")
st.caption("Query your knowledge base with zero hallucination and verified source citations.")

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if available
        if msg.get("sources"):
            st.markdown("**Sources:**")
            sources_html = "".join([f'<span class="source-tag">📄 {src}</span>' for src in msg["sources"]])
            st.markdown(sources_html, unsafe_allow_html=True)

# Chat Input & Interaction Flow
if prompt := st.chat_input("Ask a question from your uploaded documents..."):
    # Add User Message to State & Display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"query": prompt}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response generated.")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        st.markdown("**Sources:**")
                        sources_html = "".join([f'<span class="source-tag">📄 {src}</span>' for src in sources])
                        st.markdown(sources_html, unsafe_allow_html=True)
                    
                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    err_msg = f"API Error ({response.status_code}): {response.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg, "sources": []})

            except requests.exceptions.ConnectionError:
                err_msg = "❌ Connection Refused: Please ensure FastAPI backend is running on `http://127.0.0.1:8000`."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "sources": []})
            except Exception as e:
                err_msg = f"❌ Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "sources": []})