# More Retrievers in LangChain

## 1. Introduction

In Retrieval-Augmented Generation (RAG), a retriever finds relevant information from a collection of documents or an external source based on a user's query.

LangChain provides different types of retrievers. Each retriever uses a different technique to find relevant information.

**Hindi Explanation:** Retriever ka kaam user ke question ke according relevant information find karna hai. Different retrievers information ko different methods se search karte hain.

## 2. BM25 Retriever

**BM25 (Best Matching 25)** is a keyword-based information retrieval algorithm. It ranks documents according to how well their words match the user's query.

### How it works
1. Takes a user's query.
2. Matches query keywords with document keywords.
3. Calculates a relevance score for each document.
4. Returns the most relevant documents.

**Example:**

Query: `Python programming`

A document containing the words “Python” and “programming” may receive a higher relevance score.

**Hindi Explanation:** BM25 mainly keywords ke basis par documents search karta hai. Iske liye vector embeddings zaroori nahi hote.

**Best use:** Keyword-based search, especially when exact terms matter.

## 3. ParentDocumentRetriever

ParentDocumentRetriever retrieves smaller document chunks for accurate searching and then returns their larger parent documents to provide more context.

### How it works
1. Splits documents into smaller child chunks.
2. Stores the child chunks for retrieval.
3. Searches for relevant child chunks.
4. Retrieves their corresponding parent documents.

**Example:**

A large textbook chapter is divided into small chunks. If one chunk matches the query, the retriever can return the larger section containing that chunk.

**Hindi Explanation:** Chhote chunks se accurate search hota hai, aur parent document se additional context milta hai.

**Best use:** When precise retrieval and broader context are both important.

## 4. TimeWeightedVectorRetriever

TimeWeightedVectorRetriever considers both the relevance of documents and their recency. It can prioritize recently accessed or updated information, depending on its configuration.

### How it works
1. Retrieves documents based on similarity.
2. Considers recency or time-based information.
3. Applies a time-weighted relevance score.
4. Returns documents according to the resulting ranking.

**Example:**

In a personal assistant application, recently accessed notes may be preferred over older notes when their relevance is comparable.

**Hindi Explanation:** Ye retriever relevance ke saath time factor ko bhi consider kar sakta hai. Isliye recent information ko priority mil sakti hai.

**Best use:** Applications where recent interactions or information are important.

## 5. SelfQueryRetriever

SelfQueryRetriever uses a language model to convert a natural-language question into a search query with filters for document metadata.

### How it works
1. Receives a natural-language question.
2. Uses a language model to identify the search query and possible metadata filters.
3. Searches the underlying document store.
4. Returns matching documents.

**Example:**

Query: `Find science books published after 2020`

The retriever may use `science` as a search term and `publication_year > 2020` as a metadata filter, if the document store supports that metadata.

**Hindi Explanation:** Ye natural-language question ko search query aur metadata filters mein convert kar sakta hai.

**Best use:** Searching structured document collections using conditions such as date, category, or author.

## 6. EnsembleRetriever

EnsembleRetriever combines results from multiple retrievers. For example, it can combine keyword-based retrieval with vector similarity search.

### How it works
1. Sends the query to multiple retrievers.
2. Collects their results.
3. Combines and ranks the results using a configured method, such as Reciprocal Rank Fusion (RRF).
4. Returns the combined results.

**Example:**

A BM25 retriever finds documents containing exact keywords, while a vector retriever finds semantically similar documents. EnsembleRetriever can combine both sets of results.

**Hindi Explanation:** Ye multiple retrievers ki strengths ko combine karta hai, jisse different search methods se useful results mil sakte hain.

**Best use:** Applications that benefit from combining keyword and semantic search.

## 7. MultiVectorRetriever

MultiVectorRetriever allows multiple vectors or searchable representations to be associated with one original document.

### How it works
1. Creates multiple representations of a document.
2. Stores those representations in a vector store.
3. Searches the representations using vector similarity.
4. Retrieves the corresponding original document or parent document.

**Example:**

A report may have separate summaries or searchable representations for different sections, while all of them point back to the original report.

**Hindi Explanation:** Ek original document ke multiple searchable vectors ho sakte hain. Inmein se koi relevant match hone par original document retrieve kiya ja sakta hai.

**Best use:** When one document needs multiple searchable representations.

## 8. ArxivRetriever

ArxivRetriever retrieves research papers from arXiv, a repository of scientific research papers.

### How it works
1. Receives a research-related query.
2. Searches for relevant papers using the available integration.
3. Retrieves matching paper information.
4. Returns results that can be used in a research assistant or RAG pipeline.

**Example:**

Query: `Recent research papers on large language models`

It can help find relevant research papers from arXiv.

**Hindi Explanation:** Ye scientific research papers search karne ke liye useful hai.

**Best use:** Research assistants and scientific literature search.

## 9. Quick Comparison

| Retriever | Main technique | Useful for |
|---|---|---|
| BM25Retriever | Keyword matching | Exact keyword search |
| ParentDocumentRetriever | Child-chunk search with parent retrieval | Accurate search with context |
| TimeWeightedVectorRetriever | Similarity plus time weighting | Recent or time-sensitive information |
| SelfQueryRetriever | LLM-generated query and metadata filters | Filtered document search |
| EnsembleRetriever | Combines multiple retrievers | Hybrid retrieval |
| MultiVectorRetriever | Multiple representations per document | Searching summaries or multiple document views |
| ArxivRetriever | arXiv research-paper search | Scientific research |

## 10. Final Summary

Different retrievers solve different retrieval problems:

- **BM25Retriever:** Finds documents using keyword matching.
- **ParentDocumentRetriever:** Searches small chunks and can return larger parent documents.
- **TimeWeightedVectorRetriever:** Incorporates recency into retrieval.
- **SelfQueryRetriever:** Can generate metadata filters from natural-language questions.
- **EnsembleRetriever:** Combines results from multiple retrievers.
- **MultiVectorRetriever:** Supports multiple searchable representations of the same document.
- **ArxivRetriever:** Searches scientific papers available through arXiv.

**Conclusion:** The choice of retriever depends on the type of data, the search requirement, the need for context, and whether keyword matching, semantic similarity, metadata filtering, or external search is required.

---

## 11. Sample Code and Official Documentation

> **Note:** These examples use Python. Some retrievers require additional packages or a configured vector store. Install only the dependencies needed for the example you want to run.

### 11.1 BM25Retriever

**Documentation:** https://reference.langchain.com/python/langchain-community/retrievers/bm25/BM25Retriever

**Install:**

```bash
pip install -U langchain-community rank-bm25
```

**Sample code — `bm25_retriever.py`:**

```python
from langchain_community.retrievers import BM25Retriever

documents = [
    "Python is used for programming and automation.",
    "Java is an object-oriented programming language.",
    "Photosynthesis helps green plants produce food.",
]

retriever = BM25Retriever.from_texts(documents)
retriever.k = 2

query = "Python programming"
results = retriever.invoke(query)

for doc in results:
    print(doc.page_content)
```

**What to learn:** BM25 ranks the documents according to keyword relevance. It does not require embedding generation.

### 11.2 ParentDocumentRetriever

**Documentation:** https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever

**Install:**

```bash
pip install -U langchain-classic langchain-community langchain-text-splitters langchain-ollama faiss-cpu
```

**Sample code — `parent_document_retriever.py`:**

```python
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

documents = [
    Document(
        page_content=(
            "Photosynthesis is the process by which green plants "
            "make food using sunlight, water, and carbon dioxide. "
            "Chlorophyll absorbs light energy. Oxygen is released "
            "during this process. Photosynthesis supports life "
            "by producing organic food."
        )
    )
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.from_documents(
    [],
    embeddings
)

store = InMemoryStore()

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=10
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
)

retriever.add_documents(documents)

results = retriever.invoke("What does chlorophyll do?")

for doc in results:
    print(doc.page_content)
```

**What to learn:** Small child chunks are used for searching, while the corresponding parent documents provide broader context.

### 11.3 TimeWeightedVectorStoreRetriever

**Documentation:** https://reference.langchain.com/python/langchain-classic/retrievers/time_weighted_retriever/TimeWeightedVectorStoreRetriever

**Sample code — `time_weighted_retriever.py`:**

```python
from datetime import datetime, timedelta, timezone

from langchain_core.documents import Document
from langchain_classic.retrievers import (
    TimeWeightedVectorStoreRetriever
)

# In a complete application, configure a supported vector store.
# This retriever combines vector-store relevance with recency.

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=your_vectorstore,
    decay_rate=0.01,
    k=3,
)

old_date = datetime.now(timezone.utc) - timedelta(days=7)

retriever.add_documents([
    Document(
        page_content="Python is used for automation.",
        metadata={
            "last_accessed_at": old_date,
            "created_at": old_date,
        },
    ),
    Document(
        page_content="Python supports web development and AI.",
        metadata={
            "last_accessed_at": datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc),
        },
    ),
])

results = retriever.invoke("Python applications")

for doc in results:
    print(doc.page_content)
```

**Important:** Replace `your_vectorstore` with an initialized vector store compatible with the retriever. The example illustrates the API and needs that setup before it can run.

**What to learn:** The decay rate controls how quickly recency influence decreases. This retriever is particularly useful for memory-based applications.

### 11.4 SelfQueryRetriever

**Documentation:** https://reference.langchain.com/python/langchain-classic/retrievers/self_query/base/SelfQueryRetriever

**Sample code — `self_query_retriever.py`:**

```python
from langchain_core.documents import Document
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import (
    AttributeInfo
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

documents = [
    Document(
        page_content="Python basics for beginners.",
        metadata={"category": "programming", "year": 2024},
    ),
    Document(
        page_content="Advanced Java concepts.",
        metadata={"category": "programming", "year": 2022},
    ),
    Document(
        page_content="Introduction to photosynthesis.",
        metadata={"category": "science", "year": 2023},
    ),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="self_query_demo",
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

metadata_field_info = [
    AttributeInfo(
        name="category",
        description="The subject category of the document",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The publication year",
        type="integer",
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Short educational documents.",
    metadata_field_info=metadata_field_info,
    search_kwargs={"k": 3},
)

results = retriever.invoke(
    "Find programming documents published after 2023"
)

for doc in results:
    print(doc.page_content, doc.metadata)
```

**Install:**

```bash
pip install -U langchain-classic langchain-chroma langchain-ollama
```

**What to learn:** The LLM interprets the question and can generate metadata filters. Filter support depends on the vector-store integration and its translator. Ensure Ollama is running and the required models are installed.

### 11.5 EnsembleRetriever

**Documentation:** https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/ensemble.py

**Sample code — `ensemble_retriever.py`:**

```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings

documents = [
    "Python is used for automation and AI.",
    "Java is an object-oriented language.",
    "Photosynthesis converts light energy into chemical energy.",
]

bm25_retriever = BM25Retriever.from_texts(documents)
bm25_retriever.k = 2

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.from_texts(
    documents,
    embeddings
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

results = ensemble_retriever.invoke("Python and AI")

for doc in results:
    print(doc.page_content)
```

**Install:**

```bash
pip install -U langchain-classic langchain-community langchain-ollama faiss-cpu rank-bm25
```

**What to learn:** BM25 finds keyword matches, while vector search finds semantic matches. EnsembleRetriever combines their rankings using weighted rank fusion.

### 11.6 MultiVectorRetriever

**Documentation:** https://reference.langchain.com/python/langchain-classic/retrievers/multi_vector/MultiVectorRetriever

**Sample code — `multi_vector_retriever.py`:**

```python
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.from_texts(
    ["Python programming and automation"],
    embeddings,
    metadatas=[{"doc_id": "doc1"}],
)

store = InMemoryStore()

parent_document = Document(
    page_content=(
        "Python is a versatile programming language used "
        "for automation, web development, data analysis, and AI."
    ),
    metadata={"title": "Introduction to Python"},
)

store.mset([("doc1", parent_document)])

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="doc_id",
)

results = retriever.invoke("Python automation")

for doc in results:
    print(doc.page_content)
```

**What to learn:** A searchable child representation is associated with an original parent document through a shared ID. A match to the child representation can return the parent document.

### 11.7 ArxivRetriever

**Documentation:** https://reference.langchain.com/python/langchain-community/retrievers/arxiv/ArxivRetriever

**Install:**

```bash
pip install -U langchain-community arxiv
```

**Sample code — `arxiv_retriever.py`:**

```python
from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=3,
    get_full_documents=False,
)

results = retriever.invoke(
    "Retrieval augmented generation"
)

for doc in results:
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Content:", doc.page_content[:500])
    print("-" * 50)
```

**What to learn:** ArxivRetriever searches scientific papers through the arXiv integration. It requires internet access and is useful for research-focused RAG applications.

---

## 12. Important Learning Notes

1. **BM25Retriever** is a good starting point for understanding keyword-based retrieval.
2. **ParentDocumentRetriever** and **MultiVectorRetriever** teach you how to retrieve parent documents from smaller searchable representations.
3. **TimeWeightedVectorStoreRetriever** is useful for understanding recency-aware memory retrieval.
4. **SelfQueryRetriever** demonstrates how an LLM can convert natural-language requests into structured search filters.
5. **EnsembleRetriever** demonstrates how to combine different retrieval strategies.
6. **ArxivRetriever** shows how a RAG system can retrieve information from an external scientific-paper source.

**Recommended practice order:** BM25Retriever → EnsembleRetriever → ParentDocumentRetriever → MultiVectorRetriever → ArxivRetriever → SelfQueryRetriever → TimeWeightedVectorStoreRetriever.

**Version note:** LangChain APIs and package locations can change between releases. If an import fails, check the linked official reference for the installed version before changing the code.