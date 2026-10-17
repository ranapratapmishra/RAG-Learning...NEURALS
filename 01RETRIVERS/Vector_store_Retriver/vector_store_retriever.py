from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def main():
    # Step 1: Create source documents
    documents = [
        Document(
            page_content="LangChain helps developers build LLM applications easily."
        ),
        Document(
            page_content="Chroma is a vector database optimized for LLM-based search."
        ),
        Document(
            page_content="Embeddings convert text into high-dimensional vectors."
        ),
        Document(
            page_content="OpenAI provides powerful embedding models."
        ),
    ]

    # Step 2: Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step 3: Create a Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name="my_collection",
    )

    # Step 4: Convert the vector store into a retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    # Step 5: Ask a question
    query = input("Enter your question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    try:
        results = retriever.invoke(query)

        if not results:
            print("No relevant documents found.")
            return

        # Step 6: Display retrieved documents
        for index, doc in enumerate(results, start=1):
            print(f"\n--- Result {index} ---")
            print(doc.page_content)
            print(f"Metadata: {doc.metadata}")

    except Exception as error:
        print("\nAn error occurred:", error)
        print("Check your internet connection, model download, and installed packages.")


if __name__ == "__main__":
    main()