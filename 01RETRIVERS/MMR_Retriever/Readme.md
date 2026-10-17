# MMR Retriever with LangChain and FAISS

## 1. Introduction

**Maximal Marginal Relevance (MMR)** is an information retrieval technique that selects documents that are relevant to a query while reducing redundancy among the selected results.

In normal similarity search, retrieved documents may contain very similar information. MMR helps select results that provide more diverse information while remaining relevant to the query.

This project implements MMR retrieval using LangChain, FAISS, and Hugging Face embeddings.

## 2. Technologies Used

- Python
- LangChain
- FAISS
- Hugging Face Embeddings
- Sentence Transformers

## 3. Project Structure

```text
MMR_Retriever/
├── mmr_retriever.py
├── requirements.txt
└── Readme.md
```

## 4. How MMR Works

1. Create a collection of documents.
2. Convert documents into vector embeddings.
3. Store the embeddings in a FAISS vector store.
4. Retrieve candidate documents for the user's query.
5. Apply MMR to select relevant and diverse documents.
6. Display the selected results.

## 5. Why Use MMR?

Regular similarity search may return documents that repeat the same information.

MMR aims to reduce this redundancy by considering both:

- **Relevance:** How relevant a document is to the query.
- **Diversity:** How different a document is from documents already selected.

MMR is useful in RAG applications because the retrieved context can contain different but relevant information.

## 6. Installation

Open the terminal in the project directory:

```powershell
cd D:\rana_works\RAG-Learning\01RETRIVERS\MMR_Retriever
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The embedding model may download its files the first time it runs. An internet connection is required for the initial download.

## 7. Run the Project

Execute:

```powershell
python .\mmr_retriever.py
```

Enter a query when prompted. For example:

```text
Enter your question: What is LangChain?
```

The program displays up to three retrieved documents.

## 8. Important Code Explained

| Code | Purpose |
|---|---|
| `Document(page_content="...")` | Creates a text document |
| `HuggingFaceEmbeddings(...)` | Generates local text embeddings |
| `FAISS.from_documents(...)` | Creates a FAISS vector store |
| `search_type="mmr"` | Enables MMR retrieval |
| `k=3` | Requests up to three final results |
| `fetch_k=6` | Considers up to six candidate documents |
| `lambda_mult=0.5` | Balances relevance and diversity |
| `retriever.invoke(query)` | Retrieves documents for the query |

## 9. Understanding lambda_mult

The `lambda_mult` parameter controls the balance between relevance and diversity.

- `lambda_mult = 1`: Prioritizes relevance.
- `lambda_mult = 0`: Prioritizes diversity.
- `lambda_mult = 0.5`: Balances relevance and diversity.

These settings affect how MMR selects documents; they do not guarantee that every result will be unique or equally relevant.

## 10. Why FAISS?

FAISS is a library for efficient similarity search over vector representations. In this project, FAISS stores the document vectors and supports the retrieval process used by the MMR retriever.

## 11. Limitations

- The project uses six hard-coded sample documents.
- It does not load external PDF or text files.
- It retrieves documents but does not generate a final answer using an LLM.
- The FAISS index is created when the script runs; it is not saved for later reuse.

## 12. Conclusion

This project demonstrates how MMR retrieval can select relevant documents while considering diversity.

By combining LangChain, FAISS, and Hugging Face embeddings, it provides a practical foundation for understanding diverse retrieval in RAG pipelines.
