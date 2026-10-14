from io import BytesIO

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_faiss_index(uploaded_files, api_key: str):
    """Read uploaded PDFs in memory, split text, embed chunks, and create FAISS."""
    documents = []

    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        reader = PdfReader(BytesIO(pdf_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number,
                        },
                    )
                )

    if not documents:
        raise ValueError(
            "No selectable text was found in the uploaded PDFs. "
            "Scanned PDFs need OCR before they can be searched."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from the uploaded PDFs.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vectorstore, len(chunks)
