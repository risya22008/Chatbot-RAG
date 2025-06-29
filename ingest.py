import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

# Load env vars
load_dotenv()
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

# Folder tempat dokumen mentah
SOURCE_DIR = "./docs"
DB_DIR = "./db"

# Inisialisasi embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load dokumen dari folder
def load_documents(source_dir):
    documents = []
    loader = DirectoryLoader(source_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(loader.load())

    return documents

# Split teks panjang
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Simpan ke Chroma vectorstore
def save_to_vectorstore(splits):
    chroma_settings = Settings()
    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embedding,
        persist_directory=DB_DIR
    )
    vector_store.add_documents(documents=splits)
    print(f"✅ Saved {len(splits)} chunks to vectorstore at {DB_DIR} (auto-persisted)")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        raise Exception(f"{SOURCE_DIR} not found. Please create and add some files.")
    
    raw_docs = load_documents(SOURCE_DIR)
    print(f"📄 Loaded {len(raw_docs)} raw documents.")

    chunks = split_documents(raw_docs)
    print(f"🔍 Split into {len(chunks)} chunks.")

    save_to_vectorstore(chunks)
