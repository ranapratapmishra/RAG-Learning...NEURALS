# MMR Retriever — Code Explanation

## 1. Project Overview

This project demonstrates **Maximal Marginal Relevance (MMR)** retrieval using:

- **Python** — runs the program.
- **LangChain** — provides document and retriever interfaces.
- **FAISS** — stores document vectors and supports similarity search.
- **Hugging Face Embeddings** — converts text into numerical vectors using a local embedding model.

The program creates six sample documents, converts them into embeddings, builds a FAISS vector store, and uses an MMR retriever to return up to three relevant and diverse documents for a user's question.

> This script retrieves documents. It does not ask an LLM to write a final answer.

## 2. Complete Code

```python
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
```

## 3. Imports Explained

### `from langchain_core.documents import Document`

- Imports LangChain's `Document` class.
- A `Document` stores text in `page_content` and can also store extra information in `metadata`.
- In this example, metadata is not supplied, so it is normally an empty dictionary.

Example:

```python
doc = Document(page_content="Embeddings represent text as vectors.")
print(doc.page_content)
```

### `from langchain_community.vectorstores import FAISS`

- Imports the FAISS vector store integration.
- FAISS helps search vector representations efficiently.
- Here, LangChain uses FAISS to store and search the embeddings created from the sample documents.

### `from langchain_huggingface import HuggingFaceEmbeddings`

- Imports the Hugging Face embedding integration.
- The embedding model turns text into numerical vectors.
- The same embedding model is used to embed documents and user queries, so their vectors can be compared meaningfully.

## 4. `def main():`

```python
def main():
```

Defines a function named `main`. The main program steps are placed inside this function so they can be run together.

Python uses indentation to determine which statements belong to the function.

## 5. Step 1 — Create Sample Documents

```python
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
```

- `docs` is a Python list.
- Each `Document(...)` creates one document.
- `page_content` holds the text to be indexed.
- The six documents cover LangChain, Chroma, embeddings, MMR, and vector-store options.

Why use multiple documents? The retriever needs a collection of candidate texts to search through.

## 6. `try` and `except` — Handle Errors

```python
try:
    # operations that may fail
    ...
except Exception as error:
    print("\nAn error occurred:", error)
    print(
        "Check your internet connection, model download, "
        "and installed packages."
    )
```

- `try` contains operations that may fail, such as loading the embedding model or creating the vector store.
- `except` catches an exception and prints its message.
- `error` is the exception object containing details about the failure.

This helps show an error instead of letting the program stop with an unhandled exception. The printed troubleshooting hint is general; the actual error message gives more specific information.

## 7. Step 2 — Initialize the Embedding Model

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

- `HuggingFaceEmbeddings(...)` configures the embedding model.
- `model_name` identifies the model to use.
- `sentence-transformers/all-MiniLM-L6-v2` is a sentence-transformer model that converts text into numerical vectors.

**What is an embedding?**

An embedding is a list of numbers representing text. Text with similar meanings can have vectors that are close to each other in the embedding space.

The model's files may need to be downloaded the first time. After downloading, the model can generally run locally, without sending each embedding request to the OpenAI API.

## 8. Step 3 — Create a FAISS Vector Store

```python
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model,
)
```

- `FAISS.from_documents(...)` takes the document list and an embedding model.
- It generates embeddings for the documents and creates a FAISS vector store.
- `documents=docs` supplies the source documents.
- `embedding=embedding_model` supplies the model used to convert text into vectors.
- The resulting object is saved in `vectorstore`.

At this point, the vector store is ready for retrieval. This example creates it in memory each time the script runs; it does not save the FAISS index to disk.

## 9. Step 4 — Configure the MMR Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.5,
    },
)
```

### `vectorstore.as_retriever(...)`

Creates a retriever interface from the FAISS vector store.

### `search_type="mmr"`

Selects **Maximal Marginal Relevance** retrieval.

MMR tries to choose documents that are relevant to the query while avoiding unnecessary similarity among the selected results. This can provide a more varied set of context than simply returning the closest matches.

### `k: 3`

Requests up to **three final documents**. The actual number may be smaller if fewer suitable documents are available.

### `fetch_k: 6`

Considers up to six candidate documents before choosing the final MMR results. Here, there are six source documents in total.

### `lambda_mult: 0.5`

Controls the balance between relevance and diversity:

- `1.0` prioritizes relevance.
- `0.0` prioritizes diversity.
- `0.5` balances both factors.

This is a balance setting, not a guarantee that the results will be completely different or equally relevant.

## 10. Step 5 — Read the User's Query

```python
query = input("Enter your question: ").strip()
```

- `input(...)` displays a prompt and waits for the user to type.
- The entered text is stored in `query`.
- `.strip()` removes whitespace from the beginning and end.

Example:

```text
Enter your question: What is LangChain?
```

The string `What is LangChain?` becomes the query sent to the retriever.

## 11. Check for an Empty Query

```python
if not query:
    print("Please enter a question.")
    return
```

- `if not query` checks whether the query is empty after trimming whitespace.
- If it is empty, the program prints a message.
- `return` exits `main()` so no retrieval is attempted.

This prevents a blank query from being sent to the retriever.

## 12. Step 6 — Retrieve Documents

```python
results = retriever.invoke(query)
```

- `invoke(query)` runs the retriever with the user's query.
- The query is embedded using the configured embedding model.
- The vector store supplies candidates, and the MMR strategy selects the final documents.
- `results` is a list of returned `Document` objects.

The retrieved texts are not guaranteed to appear in a fixed order for every query.

### Check whether results were returned

```python
if not results:
    print("No relevant documents found.")
    return
```

If the returned list is empty, the program prints a message and exits `main()`.

## 13. Step 7 — Display the Results

```python
print("\nTop MMR results:")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print(f"Metadata: {doc.metadata}")
```

### `print("\nTop MMR results:")`

Prints a heading. `\n` adds a new line before the heading.

### `for i, doc in enumerate(results, start=1):`

- Loops over each retrieved document.
- `enumerate(...)` supplies both a number and the document.
- `start=1` makes numbering begin at 1 rather than 0.
- `i` is the result number; `doc` is the current document.

### `print(f"\n--- Result {i} ---")`

An f-string inserts the current result number into the output.

### `print(doc.page_content)`

Prints the text of the retrieved document.

### `print(f"Metadata: {doc.metadata}")`

Prints the document metadata. Because this example does not add metadata, it will normally display `{}`.

## 14. The Entry-Point Check

```python
if __name__ == "__main__":
    main()
```

- Python sets `__name__` to `"__main__"` when the file is run directly.
- If this file is run directly, `main()` starts the program.
- If the file is imported by another Python file, `main()` is not automatically called.

This is a common Python pattern for separating reusable code from program execution.

## 15. Complete Workflow Summary

1. Import the required classes.
2. Create six text documents.
3. Initialize the Hugging Face embedding model.
4. Convert the documents into vectors and build the FAISS vector store.
5. Configure an MMR retriever.
6. Ask the user for a query.
7. Retrieve up to three relevant and diverse documents.
8. Print the document text and metadata.
9. Display an error message if an operation inside the `try` block fails.

## 16. Important Limitations

- The six documents are hard-coded in the Python file.
- The FAISS index is not persisted to disk.
- The script retrieves documents; it does not generate a final answer with an LLM.
- MMR reduces redundancy as part of selection, but does not guarantee that every result is unique.
- The embedding model needs to be downloaded before it can be used locally for the first time.

## 17. Conclusion

This program demonstrates how to build an MMR retriever using LangChain, FAISS, and Hugging Face embeddings.

The key idea is to retrieve documents that are both relevant to a user's question and reasonably diverse from one another. This can help provide a more useful selection of context for a RAG pipeline.
