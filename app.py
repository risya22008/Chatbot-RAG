import streamlit as st
from dotenv import load_dotenv
import os
import traceback
from chromadb.config import Settings
from langchain_groq import ChatGroq
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain_chroma import Chroma as Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ======================
# Load environment
# ======================
load_dotenv()
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

# ======================
# Streamlit UI Config
# ======================
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot (LangChain + ChromaDB)")

qa_chain = None

with st.spinner("🔄 Memuat model dan vectorstore..."):
    try:
        # Load LLM dari Groq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

        # Load embedding dari HuggingFace (pakai CPU & stabilisasi tensor)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            # model_kwargs={"device": "cpu"},
            # encode_kwargs={"normalize_embeddings": True}
        )

        # Setup direktori vektor Chroma
        persist_directory = "./db"
        chroma_settings = Settings()

        if not os.path.exists(persist_directory):
            st.error("❌ Folder './db' tidak ditemukan. Pastikan sudah menjalankan `ingest.py`.")
            st.stop()

        vectordb = Chroma(
            collection_name="my_collection",
            embedding_function=embedding_model,
            persist_directory=persist_directory,
            # client_settings=chroma_settings
        )

        # Hitung jumlah dokumen
        doc_count = len(vectordb.get()['documents'])
        st.success(f"✅ Vectorstore berhasil dimuat dengan {doc_count} dokumen.")

        # Buat QA Chain
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    except Exception as e:
        st.error(f"❌ Gagal setup model: {e}")
        traceback.print_exc()
        st.stop()

# ======================
# UI: Ajukan Pertanyaan
# ======================
query = st.text_input("Masukkan pertanyaan kamu:", placeholder="Contoh: Bagaimana pengobatan jerawat?")

if st.button("Tanya") and query:
    if qa_chain is None:
        st.error("❌ QA Chain belum siap.")
    else:
        with st.spinner("🔎 Mencari jawaban..."):
            try:
                response = qa_chain.invoke({"query": query})
                result = response.get("result", "Tidak ada jawaban.")
                source_docs = response.get("source_documents", [])

                st.subheader("💡 Jawaban:")
                st.write(result)

                st.subheader("📚 Sumber Dokumen:")
                if source_docs:
                    for i, doc in enumerate(source_docs, 1):
                        source = doc.metadata.get("source", "Unknown source")
                        st.markdown(f"{i}. `{source}`")
                else:
                    st.write("Tidak ada sumber yang ditampilkan.")
            except Exception as e:
                traceback.print_exc()
                st.error(f"❌ Error saat menjawab: {e}")
