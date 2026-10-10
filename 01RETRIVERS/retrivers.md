# Types of Retrievers in LangChain

## 1. What Is a Retriever?

A retriever is a component in LangChain that fetches relevant documents from a data source in response to a user's query.

A retriever helps an application find useful information before passing it to an LLM.

### Basic Architecture

```text
        User Query
             |
             v
         Retriever
             |
             v
         Data Source
             |
             v
    Relevant Documents
             |
             v
             LLM
             |
             v
        Final Answer
```

**Important points:**

- There are multiple types of retrievers.
- Retrievers can work with different data sources.
- Retrievers can use different search strategies.
- LangChain retrievers implement a common runnable-style interface.

---

## 2. Types of Retrievers Based on Data Source

This classification depends on **where the information comes from**.

### 2.1 Wikipedia Retriever

A Wikipedia Retriever retrieves information from Wikipedia articles.

**Use cases:**
- General knowledge questions
- Historical information
- People, places and events
- Basic factual question answering

**Workflow:**

```text
User Query
    |
    v
Wikipedia Retriever
    |
    v
Wikipedia
    |
    v
Relevant Articles
```

**Example question:**

"What is Retrieval-Augmented Generation?"

The retriever searches Wikipedia and returns relevant article content.

**Python example:**

```python
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

documents = retriever.invoke("Artificial Intelligence")

for document in documents:
    print(document.page_content)
```

Install the required packages:

```bash
pip install langchain-community wikipedia
```

**Limitation:** Results depend on Wikipedia's available content and retrieval behavior.

### 2.2 Vector Store Retriever

A Vector Store Retriever retrieves relevant documents from a vector store using a configured search method.

The vector store generally contains embeddings associated with document chunks.

**Use cases:**
- Searching personal notes
- Searching PDFs and documentation
- Company knowledge bases
- Question answering over custom documents

**Workflow:**

```text
User Query
    |
    v
Embedding Model
    |
    v
Query Vector
    |
    v
Vector Store Search
    |
    v
Relevant Chunks
```

**Python example:**

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

documents = retriever.invoke(
    "What is RAG?"
)

for document in documents:
    print(document.page_content)
```

Here, `vector_store` represents an already-created vector store.

**Important points:**

- `k=3` requests up to three documents.
- Search quality depends on embeddings, indexed content and search configuration.
- The vector store can support different search strategies, including MMR where supported.

### 2.3 Arxiv Retriever

An Arxiv Retriever retrieves research-paper information from arXiv, an open-access repository of scientific papers.

**Use cases:**
- Reading research papers
- Studying AI and machine learning research
- Finding academic publications
- Research-oriented question answering

**Workflow:**

```text
User Query
    |
    v
Arxiv Retriever
    |
    v
arXiv Repository
    |
    v
Research Paper Information
```

**Python example:**

```python
from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2
)

documents = retriever.invoke("Retrieval-Augmented Generation")

for document in documents:
    print(document.metadata)
    print(document.page_content[:500])
```

Install the required packages:

```bash
pip install langchain-community arxiv
```

**Important:** The exact retrieved content depends on the retriever configuration and the arXiv integration's supported query behavior.

---

## 3. Types of Retrievers Based on Search Strategy

This classification depends on **how the retriever finds or selects relevant information**.

The three strategies covered here are:

1. Maximum Marginal Relevance (MMR)
2. Contextual Compression
3. Multi-Query Retrieval

### 3.1 Maximum Marginal Relevance (MMR)

MMR stands for **Maximum Marginal Relevance**.

MMR attempts to select documents that are relevant to the query while reducing unnecessary similarity or duplication among the selected results.

#### Why do we need MMR?

Ordinary similarity search may return several chunks containing nearly the same information.

MMR can help provide a more diverse set of relevant results.

**Example:**

Query: "Explain RAG applications."

Ordinary similarity search might return:
- Chunk 1: RAG in customer support
- Chunk 2: RAG in customer support
- Chunk 3: RAG in customer support

MMR may instead select:
- Chunk 1: RAG in customer support
- Chunk 2: RAG in research
- Chunk 3: RAG in document question answering

These are illustrative results; actual results depend on the indexed documents.

#### Workflow

```text
User Query
    |
    v
Vector Store
    |
    v
Candidate Documents
    |
    v
MMR Selection
    |
    v
Relevant and Diverse Results
```

#### Python example

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

documents = retriever.invoke(
    "Explain RAG applications."
)

for document in documents:
    print(document.page_content)
```

**Parameters:**

- `k`: Number of documents to return.
- `fetch_k`: Number of candidate documents considered before selecting results.
- `lambda_mult`: Controls the balance between relevance and diversity. In common LangChain vector-store implementations, lower values favor greater diversity and higher values favor relevance.

**When to use MMR:** When many retrieved chunks are similar and you want a more diverse set of results.

---

### 3.2 Contextual Compression Retriever

A Contextual Compression Retriever wraps another retriever and processes its results to keep information relevant to the user's query.

It can filter out irrelevant documents or extract relevant parts from documents, depending on the configured compressor.

#### Why do we need Contextual Compression?

A retrieved document may contain useful information along with unrelated text.

Passing the entire document to an LLM can:
- Waste tokens
- Increase prompt length
- Introduce distracting information

Contextual compression helps reduce this unnecessary context.

#### Workflow

```text
User Query
    |
    v
Base Retriever
    |
    v
Retrieved Documents
    |
    v
Document Compressor
    |
    v
Relevant Context
    |
    v
LLM
```

#### Example

Original retrieved document:

```text
RAG combines retrieval and generation.

A company may use RAG to answer questions
from its internal documentation.

The company also uses a separate system
to manage employee attendance.
```

Question:

"What is RAG used for?"

Possible compressed result:

```text
A company may use RAG to answer questions
from its internal documentation.
```

The result depends on the configured compressor.

**When to use Contextual Compression:** When retrieved documents are long or contain substantial irrelevant information.

**Important:** Contextual compression does not necessarily retrieve information from a new source. It processes the results returned by a base retriever.

---

### 3.3 Multi-Query Retriever

A Multi-Query Retriever uses an LLM to generate multiple variations of a user's original question.

It retrieves documents for those variations and combines the results, commonly removing duplicate documents.

#### Why do we need Multi-Query Retrieval?

A question can be expressed in different ways. A single query may fail to retrieve useful documents that use different terminology.

Multiple query variations can improve retrieval coverage.

#### Example

Original query:

"What is RAG?"

Possible generated queries:

1. What is Retrieval-Augmented Generation?
2. How does retrieval improve LLM responses?
3. What is the role of external knowledge in RAG?

The retriever searches using these query variations and combines the returned documents.

#### Workflow

```text
             User Query
                  |
                  v
               LLM
                  |
                  v
          Multiple Queries
           /      |      \
          v       v       v
       Search   Search   Search
          \       |       /
           v      v      v
          Combine Results
                  |
                  v
          Relevant Documents
```

#### Python example

```python
from langchain_classic.retrievers import MultiQueryRetriever

base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

documents = retriever.invoke(
    "What is RAG?"
)

for document in documents:
    print(document.page_content)
```

Here, `vector_store` must be an existing vector store and `llm` must be an initialized LangChain-compatible language model.

**When to use Multi-Query Retrieval:** When a question is ambiguous, complex, or can be described using different terms.

**Important:** Multi-query retrieval requires an LLM to generate query variations and may increase latency and model usage.

---

## 4. Comparison of Retrievers

### Based on Data Source

| Retriever | Data Source | Common Use |
|---|---|---|
| Wikipedia Retriever | Wikipedia | General knowledge |
| Vector Store Retriever | Vector store | Custom documents and notes |
| Arxiv Retriever | arXiv | Research papers |

### Based on Search Strategy

| Strategy | Main Purpose | Useful When |
|---|---|---|
| MMR | Balance relevance and diversity | Results are repetitive |
| Contextual Compression | Filter or extract relevant content | Documents contain unnecessary text |
| Multi-Query | Search through query variations | A single query misses relevant information |

---

## 5. Important Difference: Data Source vs Search Strategy

These are two different ways of classifying retrieval.

**Data source** answers:

"Where does the information come from?"

Examples:
- Wikipedia
- Vector store
- arXiv

**Search strategy** answers:

"How are relevant results selected or processed?"

Examples:
- MMR
- Contextual compression
- Multi-query retrieval

They are not mutually exclusive categories.

For example, a vector store retriever can use MMR as its search strategy. A base retriever can also be wrapped by a contextual compression retriever.

```text
                 Retriever
                     |
          +----------+----------+
          |                     |
          v                     v
      Data Source          Search Strategy
          |                     |
     Vector Store               MMR
     Wikipedia                  Multi-Query
     arXiv                      Contextual Compression
```

The diagram shows conceptual categories, not a requirement that every retriever supports every strategy.

---

## 6. Quick Revision

Remember these questions:

- **Wikipedia Retriever:** Where can I retrieve general knowledge from?
- **Vector Store Retriever:** How can I search my own indexed documents?
- **Arxiv Retriever:** How can I retrieve scientific-paper information?
- **MMR:** How can I reduce repetitive results while keeping relevant information?
- **Contextual Compression:** How can I keep only the useful parts of retrieved documents?
- **Multi-Query:** How can I search using several versions of the same question?

### Final Summary

A retriever fetches relevant documents in response to a query.

Retrievers can be classified by their data source or by their retrieval strategy. Selecting the appropriate source and strategy can improve the usefulness of the context passed to an LLM in a RAG application.
