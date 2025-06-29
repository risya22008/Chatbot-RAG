from qa_service import get_answer

if __name__ == "__main__":
    query = "Apa itu AI?"
    result, sources = get_answer(query)
    print("Jawaban:", result)
    print("Sumber:", sources)