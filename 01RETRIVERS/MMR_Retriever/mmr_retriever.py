
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    # Step 1: Create sample documents
    docs = [
        Document(
            page_content="LangChain makes it easy to work with LLMs."
        ),
        Document(
            page_content="LangChain is used to build LLM based applications."
        ),
        Document(
            page_content="Chroma is used to store and search document embeddings."
        ),
        Document(
            page_content="Embeddings are vector representations of text."
        ),
        Document(
            page_content="MMR helps you get diverse results when doing similarity search."
        ),
        Document(
            page_content="LangChain supports Chroma, FAISS, Pinecone, and more."
        ),
    ]

    try:
        # Step 2: Initialize a local embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Step 3: Create a FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model,
        )

        # Step 4: Enable MMR retrieval
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 6,
                "lambda_mult": 0.5,
            },
        )

        # Step 5: Ask the user for a query
        query = input("Enter your question: ").strip()

        if not query:
            print("Please enter a question.")
            return

        # Step 6: Retrieve diverse, relevant documents
        results = retriever.invoke(query)

        if not results:
            print("No relevant documents found.")
            return

        # Step 7: Display the results
        print("\nTop MMR results:")

        for i, doc in enumerate(results, start=1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content)
            print(f"Metadata: {doc.metadata}")

    except Exception as error:
        print("\nAn error occurred:", error)
        print(
            "Check your internet connection, model download, "
            "and installed packages."
        )


if __name__ == "__main__":
    main()
