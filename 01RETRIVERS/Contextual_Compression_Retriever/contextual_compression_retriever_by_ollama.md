# Contextual Compression Retriever using Ollama

This project uses local Ollama models instead of OpenAI, so an OpenAI API key is not needed.

## Models

- `qwen2.5:3b` — used by `ChatOllama` to extract relevant information.
- `nomic-embed-text` — used by `OllamaEmbeddings` to embed documents and queries.

## Requirements

- Ollama installed and running on your computer.
- Python 3.12 (or another compatible Python version).
- Models downloaded locally.

## Setup (Windows PowerShell)

From this project folder, run:

```powershell
ollama list
```

If either model is missing, download it:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the script:

```powershell
python .\contextual_compression_retriever.py
```

## How it works

1. `OllamaEmbeddings` turns each sample document into a vector using `nomic-embed-text`.
2. `FAISS.from_documents()` stores those vectors for similarity search.
3. The base retriever fetches up to four potentially relevant documents.
4. `ChatOllama` runs `LLMChainExtractor`, which keeps the parts relevant to the question.
5. `ContextualCompressionRetriever` returns the compressed results.

## Main imports

```python
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
```

## Troubleshooting

- **Model not found:** run the matching `ollama pull ...` command.
- **Cannot connect to Ollama:** open the Ollama desktop app. If needed, run `ollama serve` in a separate terminal.
- **Slow output:** local models may take time, especially on CPU. Wait for the first response to finish.
- **Import error:** run `python -m pip install -r requirements.txt` using the same Python interpreter used to run the script.
- **Deprecation warning:** a warning from `langchain-community` is not necessarily a runtime failure; check the final traceback line if the script stops.

This example uses a small in-memory dataset. It does not load external files or persist the FAISS index.
