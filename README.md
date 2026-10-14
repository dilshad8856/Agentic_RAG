# AI RAG Assistant — LangChain + OpenAI + FAISS + Streamlit

This app lets a client enter an OpenAI API key, connect, upload PDF files, and ask questions about those PDFs.

## Requirements

- Python 3.11 or 3.12 recommended
- An OpenAI API key with API access and billing/credits enabled
- Internet access for OpenAI API requests

## Setup on Windows (VS Code)

Open this folder in VS Code. Open Terminal and run:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\\Scripts\\activate.bat
```

Then run the install and Streamlit commands.

## Use

1. Enter the OpenAI API key in the interface.
2. Click **Connect to OpenAI**.
3. Upload one or more text-based PDFs.
4. Click **Build knowledge base**.
5. Ask questions in the chat box.

## Important behavior and security

- LangGraph is not used.
- The API key is kept in Streamlit session memory and is not written to a file by this example.
- FAISS stays in memory for the active session. The app does not save the vector index or uploaded PDFs to local disk.
- The text sent for embeddings and question answering is sent to OpenAI.
- OpenAI API usage may incur charges. The Connect action makes a small embedding request and a small chat request to test access.
- This sample does not include user accounts, rate limiting, or production-grade hosting security. If you deploy it publicly, use HTTPS, authentication, resource limits, and a security review.
- Scanned/image-only PDFs need OCR before their contents can be searched.
