# Multi-Query Retriever

## What is it?
A Multi-Query Retriever uses an LLM to rewrite one user question into several alternative questions. It retrieves documents for those variations and combines the results. This can find relevant information that a single wording might miss.

## Project structure
```text
Multi_Query_Retriever/
├── multi_query_retriever.py
├── requirements.txt
├── README.md
└── code_def.md
```

## Requirements
- Python 3.10 or newer (Python 3.12 is generally suitable).
- Internet access for installing packages and downloading the embedding model.
- An OpenAI API key for generating alternative queries. OpenAI API billing is separate from a ChatGPT subscription.
- Text embeddings run locally using `sentence-transformers/all-MiniLM-L6-v2`.

## Setup (Windows PowerShell)
Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the API key for the current PowerShell session (replace the placeholder; do not include angle brackets):

```powershell
$env:OPENAI_API_KEY="paste_your_api_key_here"
```

Run the program:

```powershell
python .\multi_query_retriever.py
```

Example question:
```text
How to improve energy levels and maintain balance?
```

## Important
- Never paste your API key into this repository or commit it to GitHub.
- The API key environment variable set this way lasts only for the current PowerShell session.
- FAISS is created in memory for this demo; the script does not save the index to disk.
