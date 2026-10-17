# Vector Store Retriever with LangChain, Chroma, and Hugging Face

## 1. Introduction

A **Vector Store Retriever** in LangChain helps search for and retrieve documents based on semantic similarity.

In this project, we use:

- **LangChain** to manage documents and retrieval.
- **Hugging Face Embeddings** to convert text into numerical vectors.
- **Chroma** to store document embeddings and perform similarity search.
- **Python** to implement the complete workflow.

This project uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model. It does not require an OpenAI API key or OpenAI API credits.

## 2. Project Structure

```text
Vector_store_Retriver/
│
├── vector_store_retriever.py
├── requirements.txt
└── Readme.md
```

## 3. Requirements

The `requirements.txt` file contains the following dependencies:

```text
langchain-core
langchain-community
langchain-chroma
langchain-huggingface
```

Install the packages using:

```powershell
python -m pip install -r requirements.txt
```

**Note:** The Hugging Face integration uses a sentence-transformer model. If running the program reports that `sentence-transformers` is missing, install it using:

```powershell
python -m pip install sentence-transformers
```

## 4. How the Project Works

The program follows this workflow:

```text
Source Documents
       |
       v
Hugging Face Embedding Model
       |
       v
Numerical Vector Embeddings
       |
       v
Chroma Vector Store
       |
       v
Retriever
       |
       v
User Query
       |
       v
Top 2 Relevant Documents
       |
       v
Display Results
```

### Step 1: Create Source Documents

The program creates four sample documents using LangChain's `Document` class.

The documents contain information about:

1. LangChain
2. Chroma
3. Embeddings
4. OpenAI embedding models

Each document stores its text in the `page_content` field.

### Step 2: Initialize the Embedding Model

The code initializes the following model:

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

An embedding model converts text into a numerical vector that represents aspects of its meaning.

The model runs locally after its files have been downloaded. An internet connection is normally required for the initial download.

### Step 3: Create a Chroma Vector Store

```python
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection",
)
```

This step creates a Chroma vector store using the sample documents and the embedding model.

Chroma stores the document embeddings so that similar documents can be retrieved later.

### Step 4: Create a Retriever

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)
```

The retriever provides an interface for searching the vector store.

Here, `k = 2` means the retriever requests up to two relevant documents for each query.

### Step 5: Accept the User's Query

```python
query = input("Enter your question: ").strip()
```

The program asks the user to enter a question. The `strip()` method removes extra whitespace from the beginning and end of the input.

If the query is empty, the program displays a message and returns without searching.

### Step 6: Retrieve Relevant Documents

```python
results = retriever.invoke(query)
```

The retriever searches for documents relevant to the query using the configured embedding model and vector store.

The program then displays the content and metadata of each returned document.

## 5. Complete Setup Instructions

### Step 1: Open the Project Folder

Open the VS Code terminal and run:

```powershell
cd D:\rana_works\RAG-Learning\01RETRIVERS\Vector_store_Retriver
```

### Step 2: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

If `sentence-transformers` is missing, run:

```powershell
python -m pip install sentence-transformers
```

### Step 3: Run the Python Program

```powershell
python .\vector_store_retriever.py
```

On the first run, the embedding model may need to download its files. Wait for the download and initialization to finish.

### Step 4: Enter a Query

For example:

```text
Enter your question: What are embeddings?
```

The program will display up to two retrieved documents.

## 6. Example Output

The following illustrates the output format. The actual retrieval order and results may vary.

```text
Enter your question: What are embeddings?

--- Result 1 ---
Embeddings convert text into high-dimensional vectors.
Metadata: {}

--- Result 2 ---
OpenAI provides powerful embedding models.
Metadata: {}
```

The example shows how the program prints each retrieved document and its metadata.

## 7. Important Code Explained

| Code | Purpose |
|---|---|
| `Document(page_content="...")` | Creates a document containing text |
| `HuggingFaceEmbeddings(...)` | Converts text into numerical embeddings |
| `Chroma.from_documents(...)` | Creates a vector store from the documents and their embeddings |
| `collection_name="my_collection"` | Specifies the Chroma collection name |
| `vectorstore.as_retriever(...)` | Creates a retriever from the vector store |
| `search_kwargs={"k": 2}` | Requests up to two retrieved documents |
| `retriever.invoke(query)` | Retrieves documents relevant to the query |
| `doc.page_content` | Accesses a document's text |
| `doc.metadata` | Accesses a document's metadata |
| `enumerate(results, start=1)` | Numbers the displayed results starting from 1 |

## 8. Why Use Hugging Face Embeddings?

The earlier version used `OpenAIEmbeddings`, which requires a valid API key and available API credits.

The updated version uses `HuggingFaceEmbeddings` with the `sentence-transformers/all-MiniLM-L6-v2` model.

Advantages:

- No OpenAI API key is required.
- No OpenAI API credits are required for embedding generation.
- Embeddings can be generated locally after the model is downloaded.
- It is suitable for learning vector stores and retrieval without relying on a paid embedding API.

**Important:** Local embedding generation still uses your computer's resources. Initial package and model downloads require internet access.

## 9. Limitations

- The project contains only four sample documents.
- Documents are defined directly in the Python file.
- The retriever requests up to two documents.
- The program retrieves documents but does not generate a final answer using an LLM.
- Persistent Chroma storage is not explicitly configured in this example.
- Retrieval quality depends on the documents, embedding model, and query.

## 10. Troubleshooting

### Error: `ModuleNotFoundError`

Install the missing dependency in the same Python environment used to run the script:

```powershell
python -m pip install -r requirements.txt
python -m pip install sentence-transformers
```

### Error: Model Download Failed

Check your internet connection and try running the script again.

### Error: Empty Query

Enter a question when the program displays:

```text
Enter your question:
```

### Error: OpenAI API Key or Credits

This version does not use `OpenAIEmbeddings`, so an OpenAI API key and OpenAI API credits are not required for the embedding step.

## 11. Learning Objectives

After completing this project, you should understand:

- What document embeddings are.
- How a Hugging Face embedding model is initialized.
- How documents are added to a Chroma vector store.
- How a retriever searches for relevant documents.
- How to use `retriever.invoke()` to retrieve documents.
- How vector retrieval fits into a RAG pipeline.

## 12. Conclusion

This project demonstrates the basic implementation of a Vector Store Retriever using LangChain, Chroma, and Hugging Face embeddings.

It converts sample documents into numerical embeddings, stores them in a vector store, and retrieves relevant documents according to a user's query.

These concepts form an important foundation for building Retrieval-Augmented Generation (RAG) applications.
