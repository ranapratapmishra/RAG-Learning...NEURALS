
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# 1. Local embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 2. Sample documents
documents = [
    Document(
        page_content=(
            "Regular exercise, walking, and proper sleep "
            "can help improve daily energy levels."
        ),
        metadata={"source": "H1", "topic": "health"}
    ),
    Document(
        page_content=(
            "A balanced diet containing fruits, vegetables, "
            "protein, and whole grains supports overall wellness."
        ),
        metadata={"source": "H2", "topic": "nutrition"}
    ),
    Document(
        page_content=(
            "Drinking enough water throughout the day "
            "helps maintain hydration and normal body functions."
        ),
        metadata={"source": "H3", "topic": "hydration"}
    ),
    Document(
        page_content=(
            "A consistent sleep schedule and a relaxing "
            "bedtime routine can support better sleep quality."
        ),
        metadata={"source": "H4", "topic": "sleep"}
    ),
    Document(
        page_content=(
            "Mindfulness, deep breathing, and short breaks "
            "may help people manage everyday stress."
        ),
        metadata={"source": "H5", "topic": "stress"}
    ),
    Document(
        page_content=(
            "Plants use sunlight, water, and carbon dioxide "
            "to produce sugars during photosynthesis."
        ),
        metadata={"source": "I1", "topic": "science"}
    ),
    Document(
        page_content=(
            "Python is a programming language often used "
            "for automation, data analysis, and AI projects."
        ),
        metadata={"source": "I2", "topic": "programming"}
    ),
]


def main():
    print("Creating local FAISS vector store...")

    # 3. Create vector store using local embeddings
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # 4. Basic retriever
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # 5. Local LLM for generating multiple queries
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    # 6. Multi-Query Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True
    )
    retriever.verbose = True

    print("\nMulti-Query Retriever is ready!")
    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

   
    try:
        print("\nGenerating alternative queries with Ollama...")
        results = retriever.invoke(question)

        print("\nRetrieved documents:")

        if not results:
            print("No matching documents found.")
            return

        # Remove duplicate documents by content
        unique_results = {}
        for doc in results:
            unique_results[doc.page_content] = doc

        for index, doc in enumerate(unique_results.values(), start=1):
            print(f"\n--- Result {index} ---")
            print("Source:", doc.metadata.get("source", "Unknown"))
            print("Topic:", doc.metadata.get("topic", "Unknown"))
            print("Content:", doc.page_content)

        print(
            "\nTotal unique documents retrieved:",
            len(unique_results)
        )

    except Exception as error:
        print(f"\nAn error occurred: {error}")
        print(
            "Check that Ollama is running and that "
            "both models are available."
        )



if __name__ == "__main__":
    main()
