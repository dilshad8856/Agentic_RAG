import streamlit as st

from ingestion import build_faiss_index
from rag_chain import answer_question, validate_openai_connection

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI RAG Assistant")
st.write("Connect your OpenAI API key, upload PDFs, and ask questions about their contents.")

# Session-only state. The key and vector index are not written to disk.
if "connected" not in st.session_state:
    st.session_state["connected"] = False
if "index_ready" not in st.session_state:
    st.session_state["index_ready"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.subheader("Connect to OpenAI")
    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        placeholder="API_KEY",
        key="api_key_input",
    )
    if st.button("Connect to OpenAI", type="primary"):
        if not api_key_input:
            st.warning("Enter your OpenAI API key first.")
        else:
            try:
                validate_openai_connection(api_key_input)
                st.session_state["api_key"] = api_key_input
                st.session_state["connected"] = True
                st.success("OpenAI connection successful.")
            except Exception:
                st.session_state["connected"] = False
                st.error(
                    "OpenAI connection failed. Check your API key and try again."
                )
    
    model_name = st.selectbox(
        "Choose model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4o", "gpt-3.5-turbo"],
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )


if st.session_state["connected"]:
    st.success("OpenAI is connected for this session.")

    st.divider()
    st.subheader("2. Upload PDF documents")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if st.button("Build knowledge base", type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            try:
                with st.spinner("Reading PDFs and building the FAISS index..."):
                    vectorstore, chunk_count = build_faiss_index(
                        uploaded_files,
                        st.session_state["api_key"],
                    )

                st.session_state["vectorstore"] = vectorstore
                st.session_state["index_ready"] = True
                st.session_state["messages"] = []
                st.success(
                    f"Knowledge base ready. Indexed {chunk_count} text chunks."
                )
            except Exception:
                st.session_state["index_ready"] = False
                st.session_state.pop("vectorstore", None)
                st.error(
                    "Could not build the knowledge base. Check that the PDFs "
                    "contain selectable text and that your API key has embedding access."
                )

    if st.session_state["index_ready"]:
        st.divider()
        st.subheader("3. Ask questions")

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source}")

        question = st.chat_input("Ask a question about your PDFs...")

        if question:
            st.session_state["messages"].append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Searching documents and preparing an answer..."):
                        result = answer_question(
                            question=question,
                            vectorstore=st.session_state["vectorstore"],
                            api_key=st.session_state["api_key"],
                        )

                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("Sources"):
                            for source in result["sources"]:
                                st.write(f"- {source}")

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except Exception:
                st.error(
                    "The answer could not be generated. Check API access and "
                    "try again. Your API key is not shown in this error message."
                )
else:
    st.info("Enter your OpenAI API key and click Connect to OpenAI to begin.")


