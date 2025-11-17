import streamlit as st
import tempfile
import os
from llm import build_vector_db, build_engines
from chromadb import PersistentClient

st.title("🎓 EduGenie_Chatbot")
st.write('“I am with you, guiding you toward your dreams.”')

uploaded_file = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# FILE UPLOAD HANDLING

if uploaded_file is not None:

    client = PersistentClient(path="db")
    try:
        client.delete_collection("edugenie_collection")
    except:
        pass

    file_name = uploaded_file.name
    ext = os.path.splitext(file_name)[1].lower()

    if ext not in [".pdf", ".docx", ".txt"]:
        st.error("Unsupported file type. Please upload PDF, DOCX, or TXT.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    vectordb = build_vector_db(file_path)
    retriever, llm = build_engines(vectordb)

    st.session_state.vectordb = vectordb
    st.session_state.retriever = retriever
    st.session_state.llm = llm

    st.success("Document processed successfully! Ask questions.")

question = st.text_input("Ask your question:")

if st.button("Go ➤"):

    if st.session_state.vectordb is None:

        edu_keywords = [
            "study", "learn", "explain", "define", "what is", "difference",
            "example", "notes", "subject", "topic", "machine learning",
            "ai", "ml", "dl", "math", "science", "engineering",
            "concept", "education", "syllabus", "chapter"
        ]

        if any(k in question.lower() for k in edu_keywords):

            if st.session_state.llm is None:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.1
                )
                st.session_state.llm = llm

            response = st.session_state.llm.invoke(question)
            answer = response.content if hasattr(response, "content") else str(response)

            st.markdown("### Answer:")
            st.markdown(answer)

        else:
            st.error("❌ This is an educational assistant. Upload a document or ask an education-related question.")

    else:

        retriever = st.session_state.retriever
        llm = st.session_state.llm

        docs = retriever.invoke(question)

        if docs:
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
            Answer the question using ONLY the following document content.
            If answer is not found, return 'NOT_FOUND'.

            DOCUMENT:
            {context}

            QUESTION:
            {question}
            """

            result = llm.invoke(prompt)
            answer = result.content if hasattr(result, "content") else str(result)

            if "NOT_FOUND" in answer:
                fallback = llm.invoke(question)
                answer = fallback.content if hasattr(fallback, "content") else str(fallback)

            st.markdown("Answer:")
            st.markdown(answer)

        else:
            fallback = llm.invoke(question)
            answer = fallback.content if hasattr(fallback, "content") else str(fallback)

            st.markdown("Answer:")
            st.markdown(answer)
