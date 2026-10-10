# Wikipedia Retriever using LangChain

## 1. Introduction

Wikipedia Retriever is a retriever provided by LangChain Community. It searches Wikipedia for information related to a user query and returns the results as LangChain Document objects.

This project demonstrates how to retrieve information from Wikipedia using Python and LangChain.

## 2. Project Structure

```text
WikiipediaRetriver/
├── wikipedia_retriever.py
├── Readme.md
└── requirements.txt
```

## 3. Workflow

1. The user enters a query.
2. Wikipedia Retriever searches Wikipedia for relevant pages.
3. The retriever returns relevant results as Document objects.
4. Each document contains page content and metadata.
5. The program displays the retrieved information.

### Workflow Diagram

```text
User Query
    |
    v
Wikipedia Retriever
    |
    v
Wikipedia Search
    |
    v
Retrieved Documents
    |
    v
Content and Metadata
    |
    v
Terminal Output
```

## 4. Technologies Used

- Python
- LangChain
- LangChain Community
- Wikipedia

## 5. Requirements

Before running this project, make sure you have:

- Python installed on your system.
- An internet connection.
- VS Code or another Python editor.
- The project files downloaded or cloned to your computer.

### Required Python Packages

- `langchain`
- `langchain-community`
- `wikipedia`

## 6. Setup Instructions

### Step 1: Open the Project Folder

Open the VS Code terminal and navigate to the project folder:

```powershell
cd D:\rana_works\RAG-Learning\01RETRIVERS\WikiipediaRetriver
```

### Step 2: Activate the Virtual Environment

If you have already created a virtual environment named `venv` in the root project folder, activate it using:

```powershell
& D:\rana_works\RAG-Learning\venv\Scripts\Activate.ps1
```

If the virtual environment is already active, skip this step.

### Step 3: Install Required Packages

You can install the packages directly using:

```powershell
python -m pip install langchain langchain-community wikipedia
```

Alternatively, install the packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

## 7. Requirements File

The `requirements.txt` file contains the Python packages required to run this project.

Create a file named `requirements.txt` in the same folder as `wikipedia_retriever.py` and add:

```text
langchain
langchain-community
wikipedia
```

## 8. How to Run the Project

Make sure your terminal is inside the `WikiipediaRetriver` folder and your virtual environment is active.

Run:

```powershell
python wikipedia_retriever.py
```

The program will ask you to enter a question.

### Example Query

```text
Enter your question: who is Maharana Pratap?
```

You can also try:

```text
the geopolitical history of India and Pakistan
```

### Expected Output

The program retrieves up to two relevant Wikipedia documents and displays their content and metadata.

The output may include:

- Document content
- Document title
- Summary
- Source URL

The actual results may vary depending on the query and Wikipedia search results.

## 9. Important Methods and Properties

| Method or Property | Purpose |
|---|---|
| `WikipediaRetriever(top_k_results=2, lang="en")` | Configures the retriever to request up to two results from English Wikipedia |
| `top_k_results=2` | Sets the maximum number of results requested |
| `lang="en"` | Uses English Wikipedia |
| `retriever.invoke(query)` | Retrieves documents for the query |
| `doc.page_content` | Contains the document text |
| `doc.metadata` | Contains metadata such as title, summary, and source URL |

## 10. How the Code Works

1. Import `WikipediaRetriever` from LangChain Community.
2. Initialize the retriever using `top_k_results=2` and `lang="en"`.
3. Accept a search query from the user.
4. Call `retriever.invoke(query)` to retrieve documents.
5. Loop through the retrieved documents.
6. Display each document's content and metadata.

## 11. Troubleshooting

### Error: `ModuleNotFoundError`

This error usually means that a required package is missing from the active Python environment.

Run:

```powershell
python -m pip install -r requirements.txt
```

Make sure VS Code is using the same Python environment where the packages were installed.

### Error: No Internet Connection

Wikipedia Retriever needs internet access to retrieve information. Check your connection and try again.

### Deprecation Warning

A deprecation warning from LangChain Community does not necessarily mean the program has failed. If the program returns documents successfully, the retrieval operation has worked.

Check future migration guidance if you decide to update the integration.

## 12. Limitations

- An internet connection is required.
- Retrieved Wikipedia pages may change over time.
- Results depend on the query and Wikipedia search results.
- The retriever returns documents; it does not generate a final answer using an LLM.
- Retrieved documents may include related pages that are not directly about the main subject.

## 13. Learning Objective

Understand how a LangChain retriever fetches external information and returns it as Document objects.

Wikipedia Retriever is one of the building blocks used in Retrieval-Augmented Generation (RAG). In a complete RAG pipeline, retrieved documents can be passed to an LLM to help generate an answer.