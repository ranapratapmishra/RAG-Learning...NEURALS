# Multi-Query Retriever — Code Explanation

This file explains `multi_query_retriever.py` in beginner-friendly language.

## 1. Imports

```python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
```

- `Document`: stores text in `page_content` and optional information in `metadata`.
- `FAISS`: builds a searchable vector index from document embeddings.
- `HuggingFaceEmbeddings`: converts text into numerical vectors using a local Hugging Face model.
- `ChatOpenAI`: connects to an OpenAI chat model, which generates alternative versions of a query.
- `MultiQueryRetriever`: creates multiple query variations with an LLM, retrieves documents for them, and combines the results.

## 2. The `main()` function

```python
def main():
```

`main()` groups the program's main steps together. The final `if __name__ == "__main__":` block calls this function when the file is run directly.

## 3. Sample documents

```python
all_docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"},
    ),
]
```

The actual script includes ten example documents. Some discuss health and wellness; others discuss unrelated topics. This lets us see which results are relevant to a query.

- `page_content`: the actual text that the retriever searches.
- `metadata`: extra details about a document. Here, `source` is a short ID such as `H1` or `I2`.

## 4. Local embeddings

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

An embedding model represents text as a list of numbers called a vector. Similar meanings tend to have vectors that are close in vector space.

The model downloads the first time it is used and then runs locally. Internet access is normally needed for the initial download. This embedding step does not require an OpenAI API key.

## 5. Create the FAISS vector store

```python
vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model,
)
```

FAISS creates an in-memory index for searching document vectors. `documents` supplies the texts and `embedding` tells FAISS how to convert them into vectors.

## 6. Regular similarity retriever

```python
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
```

This creates a standard retriever.

- `search_type="similarity"`: retrieve documents whose vectors are most similar to the query.
- `k=5`: return up to five documents.

A single query wording may not match every relevant way a topic is described.

## 7. Configure the query-generation LLM

```python
query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

This selects the chat model used to generate alternative query wordings.

- `model="gpt-4o-mini"`: the model name.
- `temperature=0`: asks for more consistent, less random outputs.

This step calls the OpenAI API, so `OPENAI_API_KEY` must be set and API access/billing must be available. ChatGPT Plus/Go and API billing are separate.

## 8. Create the Multi-Query Retriever

```python
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=query_llm,
)
```

This connects the vector-store retriever with the LLM.

Typical flow:
1. Receive the original question.
2. Ask the LLM to produce alternative ways of expressing it.
3. Search the vector store for each generated query.
4. Combine and deduplicate the retrieved documents.

The exact number and wording of generated queries are determined by the retriever's prompt/model behaviour; the script does not hard-code five query variations.

## 9. Get user input

```python
query = input("Enter your question: ").strip()
if not query:
    print("Please enter a question.")
    return
```

- `input()` reads a question typed in the terminal.
- `.strip()` removes spaces at the beginning and end.
- If the user enters nothing, the program prints a message and exits `main()` using `return`.

## 10. Run retrieval

```python
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)
```

`invoke(query)` sends the query to a retriever and returns a list of `Document` objects.

- `similarity_results`: results from one ordinary similarity search.
- `multiquery_results`: combined results from multiple generated query variations.

## 11. Print results

```python
for i, doc in enumerate(similarity_results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
```

- `enumerate(..., start=1)` numbers results from 1.
- `doc.page_content` prints the document text.
- `doc.metadata.get('source', 'unknown')` prints the source ID if it exists; otherwise it prints `unknown`.
- The separator lines make the two result sets easier to compare.

## 12. Error handling

```python
except Exception as error:
    print("\nAn error occurred:", error)
```

`try/except` catches errors so the program can show a message rather than ending with an unhandled traceback. The printed advice reminds you to check packages, internet access, and the API key. While learning, read the actual error text too; it usually contains the most specific clue.

## 13. Main guard

```python
if __name__ == "__main__":
    main()
```

Python sets `__name__` to `"__main__"` when this file is executed directly. The guard calls `main()` in that case and avoids automatically running it when the file is imported by another Python file.

## Overall flow

```text
User question
     |
     v
Regular similarity search  -----------------> Standard results
     |
     +-- Multi-Query Retriever -> LLM creates alternative queries
                                  |
                                  v
                           Search vector store
                                  |
                                  v
                         Combine/deduplicate results
                                  |
                                  v
                         Multi-query results
```

## Key difference

| Regular similarity search | Multi-Query Retriever |
|---|---|
| Searches using the original query wording | Generates alternative query wordings first |
| Usually makes one retrieval call for the query | Retrieves for multiple generated queries |
| Simpler and often faster | May find more varied relevant documents, but makes extra LLM/API calls |
