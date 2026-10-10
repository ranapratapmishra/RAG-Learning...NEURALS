# Wikipedia Retriever — LangChain

## 1. What Is a Wikipedia Retriever?

A **Wikipedia Retriever** is a retriever that queries Wikipedia to fetch relevant content for a given query.

In LangChain, `WikipediaRetriever` searches Wikipedia and returns the results as LangChain `Document` objects.

## 2. How It Works

1. You provide a query, for example, `"Albert Einstein"`.
2. The retriever sends the query to Wikipedia through its integration.
3. It retrieves relevant Wikipedia articles.
4. It returns the results as LangChain `Document` objects.

### Basic Workflow

```text
User Query
    |
    v
Wikipedia Retriever
    |
    v
Wikipedia API / Search
    |
    v
Relevant Articles
    |
    v
LangChain Document Objects
```

A returned `Document` commonly contains:

- `page_content`: the retrieved text.
- `metadata`: details such as the article title, summary, and source URL, depending on the integration and configuration.

---

## 3. Install Required Packages

The screenshots show these packages being installed in a notebook:

```python
!pip install langchain chromadb faiss-cpu openai tiktoken langchain_openai langchain-community wikipedia
```

For the Wikipedia Retriever example itself, the relevant packages are:

```bash
pip install -U langchain-community wikipedia
```

> In Google Colab or Jupyter Notebook, prefix a shell command with `!`. In PowerShell, run the command without `!`.

---

## 4. Import WikipediaRetriever

```python
from langchain_community.retrievers import WikipediaRetriever
```

This imports the Wikipedia Retriever integration from `langchain-community`.

---

## 5. Initialize the Retriever

```python
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)
```

### Parameters

- `top_k_results=2`: request up to two Wikipedia search results.
- `lang="en"`: search the English-language Wikipedia.

The exact number and content of results depend on the query and what Wikipedia returns.

---

## 6. Define a Query and Retrieve Documents

The query used in the screenshots is:

```python
query = "the geopolitical history of india and pakistan from the perspective of a chinese"
```

Send the query to the retriever:

```python
docs = retriever.invoke(query)
```

`invoke()` runs the retrieval operation and returns a list of LangChain `Document` objects.

### Complete Example

```python
from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

# Define the query
query = (
    "the geopolitical history of india and pakistan "
    "from the perspective of a chinese"
)

# Retrieve relevant Wikipedia documents
docs = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(docs, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Content:\n{doc.page_content[:500]}...")
    print(f"Metadata:\n{doc.metadata}")
```

The code prints each result's text (up to the first 500 characters) and metadata.

---

## 7. Understanding the Output

The screenshots show a result whose metadata includes the title **`United States aid to Pakistan`**, a summary, and a Wikipedia source URL.

The returned `page_content` begins with information about United States military aid and economic assistance to Pakistan. This is an example of the content returned for the query shown in the notebook.

A result is represented conceptually like this:

```python
Document(
    metadata={
        "title": "Article title",
        "summary": "Short article summary",
        "source": "Wikipedia article URL"
    },
    page_content="Retrieved article text..."
)
```

> The example above illustrates the structure. Actual metadata fields and values depend on the retrieved article and integration version.

### Access the Document Fields

```python
for doc in docs:
    print(doc.page_content)  # Retrieved text
    print(doc.metadata)      # Article details
```

- Use `doc.page_content` to read the article text.
- Use `doc.metadata` to inspect available information about the source.

---

## 8. Important Points

- A Wikipedia Retriever fetches information from Wikipedia; it does not search your private local files.
- `top_k_results` controls how many results are requested.
- `lang` selects the Wikipedia language edition.
- `invoke(query)` returns documents, not a final LLM-generated answer.
- A RAG application can pass the retrieved documents to an LLM to help answer a user's question.
- Retrieved results should be checked for relevance and accuracy.

## 9. Wikipedia Retriever in a RAG Pipeline

```text
                User Question
                      |
                      v
              Wikipedia Retriever
                      |
                      v
              Retrieved Documents
                      |
                      v
             Relevant Context + Query
                      |
                      v
                     LLM
                      |
                      v
                 Final Answer
```

## 10. Quick Revision

| Term | Meaning |
|---|---|
| Wikipedia Retriever | Retrieves relevant content from Wikipedia |
| Query | The question or search text provided to the retriever |
| `top_k_results` | Number of search results requested |
| `lang` | Language edition of Wikipedia |
| `invoke()` | Runs retrieval for a query |
| `Document` | Object containing retrieved text and metadata |
| `page_content` | Text content of the document |
| `metadata` | Additional information about the document |

**In one sentence:** A Wikipedia Retriever takes a query, searches Wikipedia, and returns relevant article content as LangChain documents.


# Document Loader vs Retriever

## 1. Introduction

In LangChain, Document Loaders and Retrievers are used to work with information, but their main purposes are different.

- **Document Loader:** Loads data from a source and converts it into Document objects.
- **Retriever:** Finds and returns documents relevant to a user query.

Both can return LangChain Document objects, but they perform different tasks.

## 2. What is a Document Loader?

A Document Loader reads data from a supported source, such as a text file or PDF, and converts it into Document objects.

For example, a `TextLoader` reads the content of a local `.txt` file.

### Example

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
```

### How it works

1. The loader receives a file path.
2. It reads the file content.
3. It converts the content into Document objects.
4. The program can access the content and metadata.

**Main purpose:** To load data from a supported source.

## 3. What is a Retriever?

A Retriever finds documents relevant to a user query and returns them as Document objects.

In this example, `WikipediaRetriever` searches Wikipedia using the user's query.

### Example

```python
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "Who is Maharana Pratap?"

docs = retriever.invoke(query)

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

### How it works

1. The user provides a query.
2. The retriever searches for relevant information.
3. It returns matching documents.
4. The program displays the content and metadata.

**Main purpose:** To retrieve documents relevant to a query.

## 4. Difference Between Document Loader and Retriever

| Feature | Document Loader | Retriever |
|---|---|---|
| Main purpose | Loads data | Retrieves relevant documents |
| Input | A supported data source, such as a file path | A user query |
| Output | Document objects | Document objects |
| Query-based retrieval | Not its usual purpose | Its main purpose |
| Example | `TextLoader` | `WikipediaRetriever` |
| Common method | `loader.load()` | `retriever.invoke(query)` |
| Example source | Local text file | Wikipedia search |

## 5. Practical Comparison

Suppose we want to learn about cricket.

### Document Loader

```python
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()
```

The loader reads the content of `cricket.txt`.

It does not automatically search Wikipedia for an answer to a question.

### Wikipedia Retriever

```python
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

docs = retriever.invoke("Who is Maharana Pratap?")
```

The retriever uses the query to find relevant Wikipedia documents.

The results depend on the query and the search results returned by Wikipedia.

## 6. Why Do Both Return Document Objects?

LangChain uses a common Document format to represent text and its associated metadata.

A Document object commonly contains:

- `page_content`: The text content.
- `metadata`: Additional information about the content, such as its source or title.

This common format makes it easier to use documents in later stages of a RAG pipeline.

## 7. Role in a RAG Pipeline

A common RAG pipeline can be represented as:

```text
Data Sources
     |
     v
Document Loader
     |
     v
Text Splitter
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
Relevant Documents
     |
     v
LLM
     |
     v
Generated Answer
```

### Explanation

1. **Document Loader:** Loads information from files or other supported sources.
2. **Text Splitter:** Divides large documents into smaller chunks.
3. **Embeddings:** Converts text chunks into numerical representations.
4. **Vector Store:** Stores and indexes the chunks for similarity search.
5. **Retriever:** Finds relevant chunks for a user query.
6. **LLM:** Uses the retrieved context to generate an answer.

This is a common RAG design. Different implementations may use different components.

### Where Does Wikipedia Retriever Fit?

`WikipediaRetriever` is a special case because it can search Wikipedia directly.

It does not require you to first load local files, create embeddings, and build a vector store for its basic search functionality.

It retrieves documents from Wikipedia based on the query.

## 8. Important Concept

**Not every Retriever requires a Vector Store.**

- `WikipediaRetriever` searches Wikipedia.
- A Vector Store Retriever searches documents indexed in a vector store.
- A Document Loader reads data from a supported source.

A Retriever and a Document Loader are therefore not interchangeable.

## 9. Limitations

### Document Loader

- It depends on the supported data source and loader implementation.
- Loading a document does not automatically mean that the answer to a query has been found.

### Wikipedia Retriever

- An internet connection is required for Wikipedia searches.
- Results depend on the query and Wikipedia search results.
- Retrieved documents may contain related information rather than a complete answer.
- Retrieval alone does not generate an LLM answer.

## 10. Quick Revision

- **Document Loader = Load data.**
- **Retriever = Find relevant documents.**
- **Document = Text content plus metadata.**
- **Text Splitter = Divide documents into chunks.**
- **Vector Store = Store and index vectors for search.**
- **LLM = Generate an answer using the available context.**

## 11. Learning Objective

Understand the difference between loading data and retrieving relevant information, and learn how both components can be used in a Retrieval-Augmented Generation (RAG) pipeline.