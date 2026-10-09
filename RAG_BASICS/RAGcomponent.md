# RAG Basic Components

## 1. Introduction to RAG Components

**RAG stands for Retrieval-Augmented Generation.**

RAG is a technique that helps a Large Language Model (LLM) answer questions using relevant information retrieved from external documents or a knowledge base.

A basic RAG system uses four important components:

1. Document Loader
2. Text Splitter
3. Vector Store
4. Retriever

Embeddings connect the text chunks to the vector store, while the LLM generates the final answer.

### Basic RAG Architecture

```text
         Documents / Files
                 |
                 v
         Document Loader
                 |
                 v
           Text Splitter
                 |
                 v
           Text Chunks
                 |
                 v
            Embeddings
                 |
                 v
            Vector Store
                 |
                 v
              Retriever
                 |
                 v
         Relevant Information
                 |
                 v
                 LLM
                 |
                 v
            Final Answer
```

**Example:** Suppose we have a text file containing information about cricket. A user asks, "Who won the Cricket World Cup in 2011?"

The RAG system loads the file, splits its content into chunks, stores their vector representations, retrieves the relevant information, and gives it to the LLM to generate an answer.

---

# 2. Document Loader

## What is a Document Loader?

A **Document Loader** is a component that loads data from different sources and converts it into a format that can be processed by a RAG application.

A source can be a text file, PDF, Word document, website, CSV file, or another supported data source.

### Why do we need a Document Loader?

Documents contain information in different formats. Before using this information in a RAG pipeline, we need to load it into our application.

A document loader helps us:

- Read data from files and other supported sources.
- Extract text from supported document formats.
- Prepare document content for further processing.
- Preserve useful metadata, such as the source file name.

### Document Loader Workflow

```text
     Source Files
          |
          v
    Document Loader
          |
          v
      Documents
          |
          v
     Text Splitter
```

### What is a Document in LangChain?

In LangChain, a `Document` object commonly contains two important fields:

- `page_content`: The actual text content.
- `metadata`: Additional information about the document, such as its source.

### Python Example: Loading a Text File

Install the required package:

```bash
pip install langchain-community
```

Create a file named `document_loader.py`:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "data/cricket.txt",
    encoding="utf-8"
)

documents = loader.load()

for document in documents:
    print("Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)
```

Create the following file:

```text
data/
└── cricket.txt
```

Example content for `cricket.txt`:

```text
Cricket is a popular sport played between two teams.
Each team usually has eleven players.
The Cricket World Cup is an international tournament.
India won the Cricket World Cup in 2011.
```

Run the program:

```bash
python document_loader.py
```

The loader reads the file and returns document objects containing the text and metadata.

### Important Points

- A loader reads data from a supported source.
- The output is commonly a list of `Document` objects in LangChain.
- The text can be passed to a text splitter.
- A loader does not automatically perform the entire RAG process.

---

# 3. Text Splitter

## What is a Text Splitter?

A **Text Splitter** divides a large document into smaller pieces called **chunks**.

These chunks are used for embedding, storage, and retrieval.

### Why do we need a Text Splitter?

Large documents may contain more information than we need for a particular question. Sending an entire document to an LLM can also consume a large context window.

Text splitting helps us:

- Break large documents into smaller pieces.
- Retrieve more focused information.
- Work with context-window limitations.
- Process large collections of documents more efficiently.

### Text Splitter Workflow

```text
       Large Document
              |
              v
         Text Splitter
              |
       +------+------+------+
       |             |      |
       v             v      v
    Chunk 1       Chunk 2  Chunk 3
```

### Important Terms

**Chunk:** A smaller piece of a document.

**Chunk Size:** The target size of each chunk, measured according to the splitter's configured unit, often characters or tokens.

**Chunk Overlap:** The amount of content shared between consecutive chunks.

### Example of Chunking

Original text:

```text
Cricket is a team sport.
Each team has eleven players.
India won the World Cup in 2011.
```

Possible chunks:

```text
Chunk 1:
Cricket is a team sport.
Each team has eleven players.

Chunk 2:
Each team has eleven players.
India won the World Cup in 2011.
```

The repeated sentence illustrates overlap. Actual chunks depend on the splitter settings.

### Python Example: Recursive Character Text Splitter

Install the package if needed:

```bash
pip install langchain-text-splitters
```

Create `text_splitter.py`:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Cricket is a popular sport.
Each team usually has eleven players.
India won the Cricket World Cup in 2011.
The tournament is played internationally.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=15
)

chunks = splitter.split_text(text)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)
```

Run the program:

```bash
python text_splitter.py
```

The program splits the text into smaller chunks. Exact chunk boundaries depend on the input and splitter settings.

### Common Text Splitters

**1. RecursiveCharacterTextSplitter**

Splits text recursively using separators, attempting to preserve meaningful text boundaries.

**2. MarkdownHeaderTextSplitter**

Splits Markdown content according to specified heading levels.

**3. Semantic Chunking**

Groups text based on semantic similarity. This approach generally requires additional logic or a suitable embedding model.

### Important Points

- Chunk size affects how much information each chunk contains.
- Overlap can help preserve context between chunks.
- Very small chunks may lose context.
- Very large chunks may retrieve unnecessary information.
- The best settings depend on the documents and application.

---

# 4. Embeddings

## What are Embeddings?

An **embedding** is a numerical vector representation of data, such as text.

An embedding model converts a text chunk into a vector that represents aspects of its meaning.

### Why do we need Embeddings?

Embeddings help a RAG system compare the meaning of text passages and user questions.

For example:

```text
"How do I repair my car?"
             |
             v
       Embedding Model
             |
             v
         Query Vector
```

A document containing "Steps to fix an automobile" may be semantically relevant even though it uses different words.

### Embedding Workflow

```text
       Text Chunk
            |
            v
      Embedding Model
            |
            v
       Number Vector
            |
            v
       Vector Store
```

### Important Points

- An embedding is a vector of numbers.
- Embeddings support semantic similarity search.
- Similarity scores help identify potentially relevant text.
- Embeddings are not the final answer generated by the LLM.
- A compatible embedding model is needed when indexing documents and embedding queries.

---

# 5. Vector Store

## What is a Vector Store?

A **Vector Store** stores vector representations of data and provides ways to search for similar vectors.

In a RAG application, a vector store commonly holds embeddings alongside their associated text chunks and metadata.

### Why do we need a Vector Store?

A knowledge base may contain hundreds or thousands of chunks. We need a way to find chunks that are relevant to a user's question.

A vector store helps us:

- Store embeddings.
- Associate vectors with document text and metadata.
- Search for similar vectors.
- Retrieve relevant document chunks.

### Vector Store Workflow

```text
        Text Chunks
             |
             v
       Embedding Model
             |
             v
          Vectors
             |
             v
        Vector Store
             |
             v
       Similarity Search
```

### Popular Vector Store Options

**FAISS:** A library for efficient similarity search over dense vectors.

**Chroma:** A vector database commonly used for local development and AI applications.

**Pinecone:** A managed vector database service.

**Qdrant:** A vector search engine and database.

**Weaviate:** A vector database supporting vector search and additional data capabilities.

The best choice depends on the project's requirements.

### Python Example: Chroma Vector Store

Install the packages:

```bash
pip install langchain-chroma langchain-huggingface sentence-transformers
```

Create `vector_store.py`:

```python
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

documents = [
    Document(
        page_content="India won the Cricket World Cup in 2011.",
        metadata={"topic": "cricket"}
    ),
    Document(
        page_content="A cricket team usually has eleven players.",
        metadata={"topic": "rules"}
    ),
    Document(
        page_content="Football is played between two teams.",
        metadata={"topic": "football"}
    )
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="rag_learning_demo"
)

results = vector_store.similarity_search(
    "When did India win the Cricket World Cup?",
    k=2
)

for document in results:
    print(document.page_content)
    print(document.metadata)
    print()
```

Run:

```bash
python vector_store.py
```

The code creates embeddings, stores the documents in Chroma, and searches for the two most similar results.

**Note:** The embedding model may need to be downloaded the first time. This example uses a local embedding model and does not require an API key for embeddings.

### Important Points

- A vector store is used for vector storage and similarity search.
- Embeddings represent the meaning of text numerically.
- `k=2` requests up to two search results.
- Similarity search does not guarantee that every returned result is correct.
- Some vector stores can persist data to disk; persistence depends on configuration.

---

# 6. Retriever

## What is a Retriever?

A **Retriever** is a component that finds and returns documents relevant to a user's query.

In a typical vector-based RAG system, the retriever uses a vector store to search for relevant chunks.

### Why do we need a Retriever?

A knowledge base may contain many documents, but the LLM usually needs only the most relevant information for the current question.

The retriever helps us:

- Search the knowledge base using a query.
- Find relevant document chunks.
- Return the selected chunks for use as context.
- Connect the retrieval stage to the generation stage.

### Retriever Workflow

```text
         User Question
                |
                v
         Query Embedding
                |
                v
            Retriever
                |
                v
           Vector Store
                |
                v
        Relevant Documents
                |
                v
                LLM
```

### Python Example: Retriever with Chroma

Create `retriever.py`:

```python
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

documents = [
    Document(
        page_content="India won the Cricket World Cup in 2011.",
        metadata={"topic": "cricket"}
    ),
    Document(
        page_content="A cricket team usually has eleven players.",
        metadata={"topic": "rules"}
    ),
    Document(
        page_content="Football is played between two teams.",
        metadata={"topic": "football"}
    )
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="retriever_learning_demo"
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

question = "When did India win the Cricket World Cup?"

results = retriever.invoke(question)

for document in results:
    print(document.page_content)
    print(document.metadata)
    print()
```

Run:

```bash
python retriever.py
```

The retriever searches the vector store and returns up to two relevant documents.

### What does `k=2` mean?

It asks the retriever to return up to two documents.

```text
k = 2  → Up to 2 documents
k = 5  → Up to 5 documents
```

Increasing `k` can provide more context, but it can also introduce irrelevant information or consume more of the LLM's context window.

### Retriever vs Vector Store

| Vector Store | Retriever |
|---|---|
| Stores and searches vectors | Provides a document-retrieval interface |
| Handles vector similarity search | Uses a configured retrieval strategy |
| Can perform similarity searches directly | Can retrieve documents from a vector store or another supported source |
| Is a storage/search component | Is a retrieval component |

A retriever does not always have to use a vector store. Some retrievers use keyword search or other methods.

---

# 7. How All Components Work Together

Suppose a user asks:

**"When did India win the Cricket World Cup?"**

The RAG system follows these steps.

### Step 1 — Document Loader

Loads the cricket text file.

### Step 2 — Text Splitter

Splits the loaded content into smaller chunks.

### Step 3 — Embeddings

Converts the chunks into numerical vectors.

### Step 4 — Vector Store

Stores the vectors with their associated document content and metadata.

### Step 5 — Retriever

Finds chunks relevant to the user's question.

### Step 6 — LLM

Receives the question and retrieved context and generates an answer.

### Complete Flow

```text
       Cricket Documents
                |
                v
        Document Loader
                |
                v
          Text Splitter
                |
                v
             Chunks
                |
                v
           Embeddings
                |
                v
           Vector Store
                |
                |        User Question
                |               |
                |               v
                |           Retriever
                |               |
                +---------------+
                        |
                        v
                Relevant Context
                        |
                        v
                       LLM
                        |
                        v
                  Final Answer
```

Example answer, if supported by the retrieved context:

```text
India won the Cricket World Cup in 2011.
```

The quality of the answer depends on the source data, retrieval quality, prompt, and LLM.

---

# 8. Quick Revision

| Component | Main Purpose | Output |
|---|---|---|
| Document Loader | Loads source information | Documents |
| Text Splitter | Breaks documents into chunks | Text chunks |
| Embedding Model | Converts text into vectors | Numerical vectors |
| Vector Store | Stores and searches vectors | Similarity-search results |
| Retriever | Finds relevant documents | Relevant chunks |
| LLM | Generates a response | Final answer |

## Easy Way to Remember

```text
LOAD
  ↓
SPLIT
  ↓
EMBED
  ↓
STORE
  ↓
RETRIEVE
  ↓
GENERATE
```

**Final summary:**

- The **Document Loader** brings data into the pipeline.
- The **Text Splitter** divides the data into manageable chunks.
- The **Embedding Model** converts text into numerical vectors.
- The **Vector Store** stores and searches those vectors.
- The **Retriever** returns relevant information for the question.
- The **LLM** uses the question and retrieved context to generate the response.

These components form the foundation of a basic Retrieval-Augmented Generation system.
