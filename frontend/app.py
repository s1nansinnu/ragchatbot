import streamlit as st
import requests
import os
from pathlib import Path

# Page configuration
st.set_page_config(page_title="RAG Chat Interface", layout="wide")
st.title("📄 RAG Chat")

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Sidebar for upload
with st.sidebar:
    st.header("📤 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if st.button("Upload PDF", use_container_width=True):
            with st.spinner("Uploading and indexing PDF..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Uploaded: {result['filename']}")
                        st.info(f"📑 Indexed {result['indexed_chunks']} chunks")
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Main chat area
st.header("💬 Ask Questions")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}")
                    st.write(f"📍 Chunk: {source['metadata'].get('chunk', 'N/A')}")
                    st.write(f"Similarity: {1 - source['distance']:.2%}")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from backend
    with st.spinner("🤖 Thinking..."):
        try:
            payload = {
                "message": prompt,
                "top_k": 3
            }
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["reply"]
                sources = result.get("source_documents", [])
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message,
                    "sources": sources
                })
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(assistant_message)
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.write(f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}")
                                st.write(f"📍 Chunk: {source['metadata'].get('chunk', 'N/A')}")
                                st.write(f"Similarity: {1 - source['distance']:.2%}")
                                st.divider()
            else:
                error_msg = f"❌ Error: {response.text}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Connection error: Make sure the backend is running at {BACKEND_URL}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
st.caption(f"🔗 Connected to backend: {BACKEND_URL}")
st.caption("💡 Tip: Upload a PDF first, then ask questions about its content!")
