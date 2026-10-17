# Vector Store Retriever with LangChain and Chroma

## 1. What is a Vector Store Retriever?

A **Vector Store Retriever** in LangChain lets you search for and fetch documents from a vector store based on semantic similarity using vector embeddings.

It is useful when documents are stored in a vector database and you want to find the documents most relevant to a user's query.

## 2. How It Works

1. Store documents in a vector store, such as Chroma or FAISS.
2. Convert each document into a numerical vector using an embedding model.
3. Convert the user's query into a vector using the same embedding model.
4. Compare the query vector with stored vectors.
5. Retrieve the top-k most similar documents.

> **Important:** The example below uses OpenAI embeddings, so you need an OpenAI API key and internet access. API usage may incur charges.

## 4. Requirements

Create `requirements.txt` in the `VectorStoreRetriever` folder:

```text
langchain-core
langchain-community
langchain-chroma
langchain-openai
```

Install packages from the project folder:

```powershell
python -m pip install -r requirements.txt
```

## 5. API Key Setup

This example uses OpenAI embeddings. In PowerShell, set your API key before running the script:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

Replace the placeholder with your own key. Never commit a real API key to GitHub.

## 6. Complete Python Example

Create `vector_store_retriever.py` and paste this code:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
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
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
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
        print("Check your API key, internet connection, and installed packages.")


if __name__ == "__main__":
    main()
```

## 7. Run the Project

Open the VS Code terminal and move into the folder. Adjust the path if you named the folder differently:

```powershell
cd D:\rana_works\RAG-Learning\02RETRIEVERS\VectorStoreRetriever
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set your API key in the same terminal session:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

Run the Python file:

```powershell
python vector_store_retriever.py
```

Example query:

```text
What is Chroma used for?
```

The program should return up to two documents semantically similar to the query. The exact order may vary.

## 8. Important Code Explained

| Code | Purpose |
|---|---|
| `Document(page_content="...")` | Creates a document containing text |
| `OpenAIEmbeddings(...)` | Converts text into embedding vectors |
| `Chroma.from_documents(...)` | Creates a Chroma vector store from documents and embeddings |
| `vectorstore.as_retriever(...)` | Exposes the vector store through a retriever interface |
| `search_kwargs={"k": 2}` | Requests up to two relevant documents |
| `retriever.invoke(query)` | Searches for documents relevant to the query |
| `doc.page_content` | Accesses the returned document text |
| `doc.metadata` | Accesses document metadata |

## 9. What Happens When You Ask a Question?

For the query `What is Chroma used for?`:

1. The embedding model converts the query into a vector.
2. Chroma compares the query vector with vectors of the stored documents.
3. The retriever returns up to two documents judged most similar to the query.
4. The program prints each document's content and metadata.

This is **semantic similarity search**, not just exact keyword matching.

## 10. Limitations and Notes

- OpenAI embeddings require an API key and internet access.
- API calls may incur charges.
- The sample documents are written directly in Python; this example does not load external files.
- The retriever returns documents but does not generate a final LLM answer.
- Chroma is created when the script runs; persistent local storage is not configured in this example.

## 11. Learning Objective

Understand how documents are embedded, stored in Chroma, and retrieved according to semantic similarity through LangChain's retriever interface.
