# ✅ FILE: qa_service.py
import os
import traceback
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_groq import ChatGroq
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain_chroma import Chroma as Chroma
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

# Initialize Hugging Face LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma vector store setup
chroma_settings = Settings()
persist_directory = "./db"

if not os.path.exists(persist_directory):
    raise Exception("Persist directory not found. Ensure to process and persist documents first.")

vectordb = Chroma(
    collection_name="my_collection",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

# Optional: Debug jumlah dokumen di vectorstore
try:
    doc_count = len(vectordb.get()['documents'])
    print(f"✅ Vectorstore loaded. Number of documents: {doc_count}")
except Exception as e:
    print("⚠️ Gagal memuat jumlah dokumen dari vectorstore:", e)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
result= retriever.invoke("Bagaimana pengobatan jerawat?")
print(result)

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def get_answer(query: str):
    """
    Processes a query through the QA chain and returns the result and sources.
    """
    try:
        response = qa_chain.invoke({"query": query})
        result = response.get("result", "")
        sources = [
            doc.metadata.get("source", "Unknown source")
            for doc in response.get("source_documents", [])
        ]
        return result, sources
    except Exception as e:
        traceback.print_exc()
        print("❌ Full error:", repr(e))
        return "Maaf, terjadi error saat menjawab.", []