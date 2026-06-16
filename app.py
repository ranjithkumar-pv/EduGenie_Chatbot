import os
import socket
import tempfile
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from llm import build_vector_db, build_engines
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def get_free_port(preferred_port=8501):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", preferred_port))
            return preferred_port
        except OSError:
            sock.bind(("0.0.0.0", 0))
            return sock.getsockname()[1]

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


def allowed_file(filename):
    extension = os.path.splitext(filename.lower())[1]
    return extension in ALLOWED_EXTENSIONS


def load_existing_vectordb():
    client = PersistentClient(path="db")
    try:
        return Chroma(client=client, collection_name="edugenie_collection", embedding_function=Embedder())
    except Exception:
        return None


def build_general_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
    )


def is_educational(question):
    edu_keywords = [
        "study", "learn", "explain", "define", "what is", "difference",
        "example", "notes", "subject", "topic", "machine learning",
        "ai", "ml", "dl", "math", "science", "engineering",
        "concept", "education", "syllabus", "chapter"
    ]
    return any(k in question.lower() for k in edu_keywords)


def query_document(question, retriever, llm):
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
            return fallback.content if hasattr(fallback, "content") else str(fallback)
        return answer
    fallback = llm.invoke(question)
    return fallback.content if hasattr(fallback, "content") else str(fallback)


@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    error = None
    question = ""
    status = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        document = request.files.get("document")

        if not question:
            error = "Please enter a question to continue."
        else:
            if document and document.filename:
                if not allowed_file(document.filename):
                    error = "Unsupported file type. Please upload PDF, DOCX, or TXT."
                else:
                    filename = secure_filename(document.filename)
                    ext = os.path.splitext(filename)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        document.save(tmp.name)
                        file_path = tmp.name

                    client = PersistentClient(path="db")
                    try:
                        client.delete_collection("edugenie_collection")
                    except Exception:
                        pass

                    vectordb = build_vector_db(file_path)
                    retriever, llm = build_engines(vectordb)
                    answer = query_document(question, retriever, llm)
                    status = f"Processed {filename} and answered from the uploaded document."
            else:
                vectordb = load_existing_vectordb()
                if vectordb:
                    retriever, llm = build_engines(vectordb)
                    answer = query_document(question, retriever, llm)
                    status = "Using the previously uploaded document."
                elif is_educational(question):
                    llm = build_general_llm()
                    response = llm.invoke(question)
                    answer = response.content if hasattr(response, "content") else str(response)
                    status = "Answered with educational AI knowledge."
                else:
                    error = "This assistant specializes in educational topics and document-based study help."

    return render_template(
        "index.html",
        answer=answer,
        error=error,
        question=question,
        status=status,
    )


if __name__ == "__main__":
    port = get_free_port(8501)
    print(f"Starting EduGenie on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
