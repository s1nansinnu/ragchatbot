# RAG Chat Frontend

A Streamlit-based web interface for the RAG Chat backend.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the backend is running:
```bash
cd ../backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Upload PDF**: Use the sidebar to upload a PDF document
2. **Ask Questions**: Type your questions in the chat input
3. **View Sources**: Expand the "Sources" section to see which document chunks were used

## Environment Variables

- `BACKEND_URL`: Backend API URL (default: `http://localhost:8000`)

To set a custom backend URL:
```bash
export BACKEND_URL=http://your-backend-url:8000
streamlit run app.py
```

## Features

- 📤 Easy PDF upload with automatic indexing
- 💬 Chat interface with conversation history
- 📚 Source document display with similarity scores
- 🔗 Real-time connection status to backend
