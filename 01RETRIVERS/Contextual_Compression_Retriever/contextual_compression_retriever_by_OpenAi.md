# Contextual Compression Retriever

## 1. Purpose

Contextual Compression Retriever improves retrieval results by keeping only the parts of retrieved documents that are relevant to the user's query.

For example, for the query **"What is photosynthesis?"**, a traditional retriever might return a full paragraph about the Grand Canyon that also contains one sentence about photosynthesis. Contextual compression can return only the relevant sentence, such as:

> Photosynthesis is the process by which green plants convert sunlight into energy.

## 2. Files

- `contextual_compression_retriever.py` — runnable example.
- `requirements.txt` — Python packages required by the example.

## 3. How the code works

### Step 1: Imports

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
```

- `Document`: stores document text and metadata.
- `FAISS`: searches for semantically similar document vectors.
- `OpenAIEmbeddings`: converts text into vectors using an OpenAI embedding model.
- `ChatOpenAI`: connects to an OpenAI chat model.
- `LLMChainExtractor`: uses an LLM to extract query-relevant parts of each retrieved document.
- `ContextualCompressionRetriever`: wraps a base retriever and a compressor.

### Step 2: Sample documents

The program creates four sample documents. Some contain photosynthesis information mixed with unrelated content about the Grand Canyon, castles, basketball, or cinema.

Each document has metadata such as `{"source": "Doc1"}`. The metadata helps identify where a result came from.

### Step 3: Create the vector store

```python
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)
```

The embedding model converts document text into vectors, and FAISS stores those vectors for similarity search.

**Important:** `OpenAIEmbeddings()` needs a valid `OPENAI_API_KEY`. OpenAI API billing/credits are separate from a ChatGPT subscription.

### Step 4: Create the base retriever

```python
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
```

`k=5` requests up to five matching documents from the base retriever. Since this example has only four documents, it cannot return five distinct documents from this sample list.

### Step 5: Set up the compressor

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
```

The LLM is used to extract the text that matters for the query. `temperature=0` is intended to make the output more consistent, but does not guarantee identical output in every situation.

### Step 6: Create the compression retriever

```python
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)
```

This combines two stages:

1. The base retriever finds potentially relevant documents.
2. The compressor processes those documents and keeps query-relevant content.

### Step 7: Ask a question

```python
query = input('Enter your question: ').strip()
compressed_results = compression_retriever.invoke(query)
```

The user enters a question, for example `What is photosynthesis?`. The `invoke()` call runs retrieval and compression.

### Step 8: Print results

The code prints each result's source metadata and compressed text. If no results are returned, it prints a message instead.

## 4. Workflow diagram

```text
User query
    |
    v
FAISS Base Retriever
    |
    v
Top matching documents
    |
    v
LLMChainExtractor (compressor)
    |
    v
Keep query-relevant text
    |
    v
Print compressed results
```

## 5. When to use it

Use contextual compression when:
- Documents are long and contain mixed information.
- Retrieved paragraphs include irrelevant sentences.
- You want to reduce unnecessary context passed to a downstream LLM.
- You want to improve the usefulness of retrieved context in a RAG pipeline.

Compression does not guarantee a correct final answer. Always check that the extracted content is relevant and preserves the needed meaning.

## 6. Setup on Windows / PowerShell

Open PowerShell in the project folder, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set your API key for the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

Replace the placeholder with your own key. Do not commit or share your real API key.

Run the script:

```powershell
python .\contextual_compression_retriever.py
```

Example query:

```text
What is photosynthesis?
```

## 7. Troubleshooting

- **Missing API key:** set `OPENAI_API_KEY` in the same terminal before running the script.
- **Insufficient quota / billing error:** check your OpenAI API billing and available credits. ChatGPT subscription and API billing are separate.
- **Package/import error:** run `python -m pip install -r requirements.txt` in the Python environment used to run the script.
- **Unexpected output:** an LLM's extraction can vary. Review the returned text rather than assuming every result will be identical to the screenshot.

## 8. Quick revision

| Component | Responsibility |
|---|---|
| FAISS | Finds candidate documents by vector similarity |
| Base retriever | Retrieves the top matching documents |
| LLMChainExtractor | Extracts text relevant to the query |
| ContextualCompressionRetriever | Combines retrieval and compression |
| `invoke(query)` | Runs the complete retrieval-compression pipeline |

**In one sentence:** Contextual Compression Retriever first retrieves documents and then compresses them to retain the content relevant to the user's question.
