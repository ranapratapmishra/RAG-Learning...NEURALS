# Compare Retrievers --- Code Explanation

## 1. Project ka purpose

Is project mein do retrieval approaches compare hoti hain:

1.  **Normal Retriever** --- original question ko directly FAISS vector
    store mein search karta hai.
2.  **Multi-Query Retriever** --- Ollama ke local LLM se original
    question ke alternative versions banwata hai. Phir original question
    aur generated queries se documents search karke results combine
    karta hai.

Dono same documents, same FAISS vector store aur same embedding model
(`nomic-embed-text`) use karte hain.

> **Note:** `compare_retrievers.py` mein Multi-Query logic manually
> implement kiya gaya hai. Ye LangChain ke `MultiQueryRetriever` class
> ko directly call nahi karta, lekin multiple queries se retrieval karne
> ka core idea follow karta hai.

## 2. Folder structure

``` text
Multi_Query_Retriever/
├── multi_query_retriever.py
├── compare_retrievers.py
├── requirements.txt
├── README.md
└── compare.md
```

-   `compare_retrievers.py` --- comparison program.
-   `requirements.txt` --- Python dependencies.
-   `README.md` --- project overview aur run instructions.
-   `compare.md` --- code ki explanation.

## 3. Imports

``` python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
```

-   **Document:** text aur metadata ko ek object mein rakhta hai.
-   **FAISS:** embeddings ke basis par similar documents search karta
    hai.
-   **ChatOllama:** local LLM se alternative search queries generate
    karwata hai.
-   **OllamaEmbeddings:** text ko numerical vectors mein convert karta
    hai.

## 4. Local models configure karna

``` python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)
```

-   `nomic-embed-text` documents aur queries ki embeddings banata hai.
-   `qwen2.5:3b` alternative search queries generate karta hai.
-   `temperature=0` generation ko comparatively consistent rakhne ki
    koshish karta hai.

Ollama models pehle download karo:

``` powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

OpenAI API key ki zaroorat nahi hai.

## 5. Documents aur metadata

Example:

``` python
Document(
    page_content="Regular exercise, walking, and proper sleep can help improve daily energy levels.",
    metadata={"source": "H1", "topic": "health"}
)
```

-   `page_content`: actual text jise search karna hai.
-   `metadata`: extra information, jaise source ID aur topic.

Example documents health, nutrition, hydration, sleep, stress, science
aur programming se related hain. Dono retrievers same documents use
karte hain.

## 6. Duplicate documents remove karna

``` python
def remove_duplicates(docs):
    unique_docs = {}

    for doc in docs:
        unique_docs[doc.page_content] = doc

    return list(unique_docs.values())
```

Multi-Query approach mein same document multiple queries se mil sakta
hai. Ye function `page_content` ko dictionary key banata hai aur
duplicate content ko ek entry mein reduce karta hai.

**Limitation:** Agar do documents ka content same ho lekin metadata alag
ho, to ek hi entry bachegi.

## 7. Normal Retriever

``` python
def run_normal_retriever(retriever, question):
    results = retriever.invoke(question)
    results = remove_duplicates(results)
```

Normal retriever original question ko directly vector store mein bhejta
hai.

Base retriever is tarah configure hai:

``` python
base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
```

`k=3` ka matlab har search mein maximum 3 matching documents return
karne ki request hai. Results mein source, topic aur document content
print hote hain. `metadata.get("source", "Unknown")` field missing hone
par `"Unknown"` deta hai.

## 8. Multi-Query Retriever

### 8.1 Alternative queries generate karna

``` python
prompt = f"""
You are a search-query generation assistant.

Generate exactly 3 different search queries that preserve
the meaning of the user's question but explore different
wordings or aspects.

Return only the queries, one per line.
Do not answer the question.
Do not add explanations.

Original question: {question}
"""

response = llm.invoke(prompt)
response_text = response.content
```

Prompt LLM ko original question ke 3 alternative search queries banane
ko kehta hai. `llm.invoke()` prompt model ko bhejta hai aur
`response.content` generated text deta hai.

### 8.2 Generated queries parse karna

``` python
generated_queries = []

for line in str(response_text).splitlines():
    query = line.strip()
    query = query.lstrip("0123456789.):-* ").strip()

    if query and query.lower() not in [
        q.lower() for q in generated_queries
    ]:
        generated_queries.append(query)
```

1.  Response ko lines mein divide kiya jaata hai.
2.  `strip()` extra spaces hataata hai.
3.  `lstrip()` start se numbering aur common bullet characters hataata
    hai.
4.  Empty lines aur repeated queries skip hoti hain.

Ye simple parser hai; unusual formatting ko har case mein perfectly
parse karna guaranteed nahi hai.

### 8.3 Original question include karna

``` python
queries = [question]

for query in generated_queries:
    if query.lower() != question.lower():
        queries.append(query)
```

Original question bhi search hota hai. Isliye retrieval sirf generated
alternatives par depend nahi karta.

### 8.4 Har query se documents search karna

``` python
all_results = []

for query in queries:
    query_results = retriever.invoke(query)
    all_results.extend(query_results)
```

Har query ko same base retriever diya jaata hai aur results
`all_results` mein add hote hain. Agar 4 queries use hui aur har search
3 documents return kare, to deduplication se pehle maximum 12 entries aa
sakti hain.

### 8.5 Results combine karna

``` python
results = remove_duplicates(all_results)
```

Sab queries ke results combine hote hain aur duplicate content remove
hota hai. Zyada documents milna automatically better accuracy ka proof
nahi hai; relevance bhi check karni chahiye.

## 9. FAISS vector store banana

``` python
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
```

Is step mein documents ki embeddings banti hain aur FAISS unhe
similarity search ke liye store karta hai. Uske baad base retriever
banta hai:

``` python
base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
```

Dono retrieval methods isi vector store aur base retriever ka use karte
hain.

## 10. User se question lena

``` python
question = input(
    "\nEnter your question: "
).strip()

if not question:
    print("Please enter a question.")
    return
```

`input()` terminal mein question leta hai, `strip()` extra spaces remove
karta hai aur blank question hone par function return kar deta hai.

Example:

``` text
How to improve energy levels and maintain balance?
```

## 11. Dono retrievers run karna

``` python
normal_results = run_normal_retriever(
    base_retriever,
    question
)

multi_results, queries = run_multi_query_retriever(
    base_retriever,
    question
)
```

Pehle normal retriever run hota hai. Phir Multi-Query method run hota
hai. Multi-Query function `multi_results` aur used queries ki list
return karta hai.

## 12. Final comparison

``` python
normal_sources = {
    doc.page_content for doc in normal_results
}

multi_sources = {
    doc.page_content for doc in multi_results
}

common_docs = normal_sources & multi_sources
extra_docs = multi_sources - normal_sources
```

-   `normal_sources`: normal retriever ke document contents.
-   `multi_sources`: Multi-Query method ke document contents.
-   `&`: intersection; dono methods mein common documents.
-   `-`: difference; Multi-Query mein mile, lekin normal retriever mein
    nahi mile documents.

Final summary documents ki counts, common documents, additional
documents aur generated alternative queries ki count print karti hai. Ye
counts retrieval quality ka direct measurement nahi hain.

## 13. Error handling

``` python
try:
    # Retrieval and comparison
    ...
except Exception as error:
    print("\nAn error occurred:", error)
```

Agar retrieval ke dauran error aaye, `except` block error message print
karta hai. Ollama connection ya model availability issue ho to Ollama
app aur downloaded models check karo.

## 14. Program run karna

PowerShell mein:

``` powershell
cd D:\rana_works\RAG-Learning\01RETRIVERS\Multi_Query_Retriever
ollama list
python -m pip install -r requirements.txt
python .\compare_retrievers.py
```

Agar models missing hon:

``` powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## 15. Normal vs Multi-Query --- summary

  Feature                Normal Retriever     Multi-Query method
  ---------------------- -------------------- --------------------------------
  Search input           Original question    Original + alternative queries
  Query-generation LLM   Nahi                 Haan, local Ollama
  Embedding model        `nomic-embed-text`   Same model
  Vector store           FAISS                Same FAISS
  Number of searches     Ek                   Multiple
  Results merge          Nahi                 Haan
  Duplicate removal      Included             Included
  OpenAI API credits     Nahi                 Nahi

## Conclusion

Normal Retriever simple, direct search ke liye useful hai. Multi-Query
approach alternate wordings use karti hai, jisse aise relevant documents
mil sakte hain jo original wording se miss ho gaye the.

Multi-Query method mein extra LLM generation aur multiple searches hote
hain, isliye woh slower ho sakta hai. Kaunsa method better hai, ye
actual questions aur retrieved documents ki relevance par depend karta
hai.
