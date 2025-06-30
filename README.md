# 🧠 RAG Chatbot API - Sistem Tanya Jawab Penyakit Kulit

**RAG Chatbot API** adalah sebuah sistem tanya jawab (*Question Answering*) yang dibangun menggunakan arsitektur **Retrieval-Augmented Generation (RAG)**. Proyek ini dirancang khusus untuk memberikan jawaban informatif atas pertanyaan-pertanyaan seputar **penyakit kulit** berdasarkan koleksi dokumen yang relevan.

Sistem ini diimplementasikan sebagai REST API menggunakan **FastAPI** dan mengintegrasikan beberapa teknologi canggih, termasuk **Hugging Face Embeddings** untuk pemahaman teks, **ChromaDB** sebagai *vector store*, dan model bahasa besar **LLaMA-3.3 70B** yang diakses melalui **Groq** untuk kecepatan inferensi yang tinggi.

---

## 🚀 Fitur Utama

-   **Tanya Jawab Bahasa Alami**: Menerima dan memproses pertanyaan dari pengguna dalam format bahasa natural.
-   **Berbasis Dokumen**: Mencari dan mengekstrak informasi relevan dari sekumpulan dokumen teks tentang penyakit kulit.
-   **Akurasi Tinggi dengan RAG**: Menggunakan metode *Retrieval-Augmented Generation* untuk menghasilkan jawaban yang lebih akurat dan kontekstual.
-   **Inferensi Cepat**: Didukung oleh `ChatGroq` (LLaMA-3.3 70B) untuk respons yang cepat dan efisien.
-   **Penyimpanan Vektor Efisien**: Menggunakan ChromaDB untuk menyimpan dan mengambil *embeddings* dokumen secara persisten.
-   **Konfigurasi Aman**: Mendukung penggunaan file `.env` untuk mengelola kunci API secara aman.

---

## 🧱 Arsitektur Sistem

Alur kerja sistem ini sederhana dan efisien, dimulai dari permintaan pengguna hingga pengiriman jawaban yang dihasilkan oleh LLM.

```
Pengguna ───▶ FastAPI (main.py)
   │
   └──▶ get_answer() [qa_service.py]
           ├── Retriever: Chroma VectorStore (mengambil dokumen relevan)
           ├── LLM: LLaMA 3.3 via Groq (menghasilkan jawaban)
           └── Embeddings: all-MiniLM-L6-v2 (mengubah teks menjadi vektor)
```

---

## 📦 Instalasi dan Penggunaan

Ikuti langkah-langkah di bawah ini untuk menjalankan API di lingkungan lokal Anda.

### 1. Kloning Repositori
Pertama, kloning repositori ini ke mesin lokal Anda:
```bash
git clone [https://github.com/username/rag-skin-chatbot-api.git](https://github.com/username/rag-skin-chatbot-api.git)
cd rag-skin-chatbot-api
```

### 2. Instalasi Dependensi
Instal semua pustaka Python yang dibutuhkan menggunakan `pip` dan file `requirements.txt`.
```bash
pip install -r requirements.txt
```
Contoh isi file `requirements.txt`:
```nginx
fastapi
uvicorn[standard]
python-dotenv
langchain
langchain-community
langchain-huggingface
langchain-chroma
langchain-groq
chromadb
sentence-transformers
```

### 3. Siapkan File `.env`
Buat sebuah file bernama `.env` di direktori utama proyek. File ini akan digunakan untuk menyimpan kunci API Anda.
```ini
GROQ_API_KEY="gsk_YourGroqApiKeyHere"
```
Ganti `gsk_YourGroqApiKeyHere` dengan kunci API Groq Anda yang valid.

### 4. Menyiapkan Dokumen dan Vector Store
a. **Tambahkan Dokumen**: Tempatkan file-file teks (`.txt`) yang berisi informasi tentang penyakit kulit ke dalam folder `docs/`.

b. **Jalankan Skrip Persiapan**: Eksekusi skrip berikut untuk memproses dokumen, membuat *embeddings*, dan menyimpannya ke dalam ChromaDB.
```bash
python prepare_vectorstore.py
```
Proses ini akan membaca semua file di `docs/`, membaginya menjadi potongan-potongan kecil (*chunks*), dan menyimpan representasi vektornya di dalam folder `db/`.

### 5. Menjalankan API
Setelah semua persiapan selesai, jalankan server API menggunakan Uvicorn:
```bash
uvicorn main:app --reload
```
API akan aktif dan dapat diakses di `http://127.0.0.1:8000`.

---

## 📡 Endpoint API

### `POST /ask`
Endpoint utama untuk mengirimkan pertanyaan dan mendapatkan jawaban.

-   **Contoh Request**:
    ```json
    {
      "query": "Bagaimana cara mengobati jerawat yang meradang?"
    }
    ```

-   **Contoh Response**:
    ```json
    {
      "result": "Jerawat yang meradang dapat diobati dengan menggunakan krim topikal yang mengandung benzoil peroksida atau asam salisilat, serta menjaga kebersihan wajah dengan mencucinya dua kali sehari. Hindari memencet jerawat untuk mencegah infeksi lebih lanjut.",
      "sources": [
        "docs/acne.txt"
      ]
    }
    ```

### `GET /status`
Endpoint untuk memeriksa apakah layanan API sedang berjalan.

-   **Contoh Response**:
    ```json
    {
      "status": "API is running",
      "service": "RAG Chatbot for Skin Diseases"
    }
    ```

---

## 📂 Struktur Folder Proyek

```
.
├── main.py                  # Titik masuk aplikasi FastAPI
├── qa_service.py            # Logika inti untuk layanan Q&A (RAG)
├── prepare_vectorstore.py   # Skrip untuk preprocessing dan vektorisasi dokumen
├── docs/                      # Folder untuk menyimpan dokumen sumber (.txt)
│   ├── acne.txt
│   └── psoriasis.txt
├── db/                        # Folder untuk menyimpan vector store Chroma (persisten)
├── .env                       # File konfigurasi untuk kunci API
├── requirements.txt           # Daftar dependensi Python
└── README.md                  # Dokumentasi proyek
```

---

## ⚠️ Catatan Penting

-   **Bukan Nasihat Medis**: Aplikasi ini adalah prototipe edukatif dan **tidak dimaksudkan untuk menggantikan diagnosis atau nasihat medis profesional**.
-   **Tujuan Proyek**: Tujuan utama proyek ini adalah untuk mendemonstrasikan penerapan arsitektur RAG pada data spesifik (medis) dan bukan untuk menciptakan alat diagnostik.
-   **Akurasi Jawaban**: Jawaban yang dihasilkan bersifat informatif dan didasarkan pada data yang diberikan. Akurasi klinis tidak dijamin. Selalu konsultasikan dengan dokter atau tenaga medis profesional untuk masalah kesehatan.
