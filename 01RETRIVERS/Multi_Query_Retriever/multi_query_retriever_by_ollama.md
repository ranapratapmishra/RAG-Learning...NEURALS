# Multi-Query Retriever Using Ollama — Code Explanation

## 1. Is project ka purpose

Is program mein hum **Multi-Query Retriever** use karte hain. User ek question deta hai, aur local Ollama LLM us question ke alternative search queries generate karta hai. Retriever in queries ki help se FAISS vector store se relevant documents retrieve karta hai.

Is project mein do local models use ho rahe hain:

- **`nomic-embed-text`** — documents aur questions ko embeddings (numerical vectors) mein convert karta hai.
- **`qwen2.5:3b`** — original question ke alternative queries generate karta hai.

Dono models Ollama ke through local machine par run hote hain. Is code mein OpenAI API key ki zaroorat nahi hai.

---

## 2. Required packages

Code mein ye imports use hue hain:

```python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
```

| Import | Purpose |
|---|---|
| `Document` | Text ko document object mein rakhne ke liye |
| `FAISS` | Embeddings ke basis par similar documents search karne ke liye |
| `ChatOllama` | Ollama ke local chat model ko call karne ke liye |
| `OllamaEmbeddings` | Ollama ke embedding model se vectors banane ke liye |
| `MultiQueryRetriever` | Ek question se multiple queries banakar retrieval karne ke liye |

---

## 3. Local embedding model

```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
```

- `OllamaEmbeddings` LangChain ko Ollama embedding model se connect karta hai.
- `nomic-embed-text` text ko numerical vector mein convert karta hai.
- FAISS in vectors ka use karke query se milte-julte documents search karta hai.

**Simple example:** “daily energy improve kaise karein?” aur “energy levels badhane ke tareeke” ke words alag hain, lekin meaning similar ho sakta hai. Embeddings semantic similarity identify karne mein madad karte hain.

> Note: Embeddings khud final answer generate nahi karte. Ye text ko search-friendly numerical representation mein convert karte hain.

---

## 4. Sample documents

Code mein `documents` list ke andar 7 sample documents hain. Har document mein do main parts hain:

```python
Document(
    page_content="Document ka actual text",
    metadata={"source": "H1", "topic": "health"}
)
```

- **`page_content`**: actual text jo retrieve ho sakta hai.
- **`metadata`**: document ke baare mein additional information, jaise source ID aur topic.

| Source | Topic | Content ka subject |
|---|---|---|
| H1 | health | Exercise, walking aur sleep |
| H2 | nutrition | Balanced diet |
| H3 | hydration | Paani aur hydration |
| H4 | sleep | Sleep schedule aur bedtime routine |
| H5 | stress | Mindfulness aur stress management |
| I1 | science | Photosynthesis |
| I2 | programming | Python aur AI projects |

Health-related documents ke saath science aur programming ke documents bhi rakhe gaye hain. Isse different questions par retrieval ka behaviour test kiya ja sakta hai.

---

## 5. `main()` function

```python
def main():
```

Program ka main workflow is function ke andar hai. Python file ke end par `main()` call hone par ye steps execute hote hain.

---

## 6. FAISS vector store create karna

```python
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
```

Is step mein:

1. `documents` list ke texts liye jaate hain.
2. `nomic-embed-text` har document ke liye embedding banata hai.
3. FAISS documents aur unke vectors ko search ke liye store karta hai.

```python
base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
```

- `as_retriever()` vector store ko retriever interface mein use karne deta hai.
- `k: 3` ka matlab hai ki har individual query ke liye retriever **top 3 matching documents** maangta hai.

Multi-Query process mein multiple queries run hoti hain, isliye final unique documents ki total count zaroori nahi ki 3 hi ho. Duplicate results ko merge karne ke baad count badal sakti hai.

---

## 7. Local LLM configure karna

```python
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)
```

- `ChatOllama` local Ollama chat model ko LangChain ke saath use karta hai.
- `qwen2.5:3b` query generation ke liye LLM hai.
- `temperature=0` output ko zyada consistent banane ke liye use hota hai; phir bhi output har situation mein bilkul identical hone ki guarantee nahi hai.

Ye model original question ka answer dene ke bajaye Multi-Query Retriever ke liye alternate search queries banane mein use hota hai.

---

## 8. Multi-Query Retriever banana

```python
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True
)
retriever.verbose = True
```

### `MultiQueryRetriever.from_llm(...)`

Isse Multi-Query Retriever configure hota hai:

- `retriever=base_retriever`: har generated query ko search karne ke liye base retriever use hoga.
- `llm=llm`: local Qwen model alternative queries generate karega.
- `include_original=True`: original question bhi retrieval queries mein include rahega, alternative queries ke saath.

### `retriever.verbose = True`

Ye setting logs/extra information dikhane mein madad kar sakti hai, taaki query generation aur retrieval process ko samajhna aasaan ho. Exact log details installed LangChain version par depend kar sakti hain.

---

## 9. User se question lena

```python
question = input("\nEnter your question: ").strip()

if not question:
    print("Please enter a question.")
    return
```

- `input()` user se question leta hai.
- `.strip()` shuru aur end ke extra spaces hata deta hai.
- Agar user blank input deta hai, to message print hota hai aur function `return` se end ho jaata hai.

Example question:

```text
How can I improve my daily energy levels?
```

---

## 10. Multi-Query retrieval chalana

```python
print("\nGenerating alternative queries with Ollama...")
results = retriever.invoke(question)
```

`retriever.invoke(question)` main retrieval operation hai. High-level process:

1. Original question Multi-Query Retriever ko diya jaata hai.
2. LLM uske liye alternative queries generate karta hai.
3. Original query bhi include hoti hai, kyunki `include_original=True` set hai.
4. Base retriever har query par FAISS search karta hai.
5. Retrieved documents ko ek combined result list mein diya jaata hai.

**Important:** Code mein query generation aur retrieval ko alag-alag manually implement nahi kiya gaya; `MultiQueryRetriever` ye workflow handle karta hai.

---

## 11. Results empty hain ya nahi

```python
if not results:
    print("No matching documents found.")
    return
```

Agar retriever koi document return nahi karta, to message print hota hai aur function aage execute nahi hota.

---

## 12. Duplicate documents remove karna

```python
unique_results = {}
for doc in results:
    unique_results[doc.page_content] = doc
```

Different queries ek hi document ko retrieve kar sakti hain. Dictionary ka use karke same `page_content` wale duplicate results ko ek entry mein rakha jaata hai.

- Dictionary key: document ka actual text (`page_content`)
- Dictionary value: document object

Agar same text dobara aaye, to same key overwrite hoti hai. Isliye final dictionary mein har unique text ki ek entry hoti hai.

**Note:** Is code mein deduplication text ke basis par hoti hai. Agar do documents ka text same hai lekin metadata alag hai, to unhe alag entries ke roop mein preserve nahi kiya jaata.

---

## 13. Documents print karna

```python
for index, doc in enumerate(unique_results.values(), start=1):
    print(f"\n--- Result {index} ---")
    print("Source:", doc.metadata.get("source", "Unknown"))
    print("Topic:", doc.metadata.get("topic", "Unknown"))
    print("Content:", doc.page_content)
```

- `enumerate(..., start=1)` results ko 1 se numbering deta hai.
- `doc.metadata.get("source", "Unknown")` source metadata print karta hai. Key missing hone par `"Unknown"` print hota hai.
- `doc.metadata.get("topic", "Unknown")` topic print karta hai.
- `doc.page_content` document ka actual text print karta hai.

Example output ka format:

```text
--- Result 1 ---
Source: H1
Topic: health
Content: Regular exercise, walking, and proper sleep can help improve daily energy levels.
```

Actual results user ke question aur model-generated queries par depend karenge.

---

## 14. Total unique documents count

```python
print(
    "\nTotal unique documents retrieved:",
    len(unique_results)
)
```

`len(unique_results)` duplicate removal ke baad unique document texts ki count batata hai.

Ye count `k=3` ke equal hona zaroori nahi hai, kyunki Multi-Query Retriever multiple queries par search karta hai aur unke results combine karta hai.

---

## 15. Error handling

```python
except Exception as error:
    print(f"\nAn error occurred: {error}")
    print(
        "Check that Ollama is running and that "
        "both models are available."
    )
```

Agar `try` block ke andar error aata hai, to program error message print karta hai. Common checks:

1. Ollama installed hai aur service run kar rahi hai.
2. Dono models download hain: `nomic-embed-text` aur `qwen2.5:3b`.
3. Required Python packages current environment mein installed hain.

Ollama models check karne ke liye PowerShell mein:

```powershell
ollama list
```

Agar koi model missing ho, to:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

---

## 16. Program ko start karna

```python
if __name__ == "__main__":
    main()
```

Python file ko directly run karne par `main()` execute hota hai. Agar file kisi doosri Python file mein import hoti hai, to ye block automatically execute nahi hota.

---

## 17. Run karne ke steps

PowerShell mein project folder par jaayein:

```powershell
cd D:\rana_works\RAG-Learning\01RETRIVERS\Multi_Query_Retriever
```

Agar `requirements.txt` mein required dependencies listed hain, unhe install karein:

```powershell
python -m pip install -r requirements.txt
```

Phir apni Python file run karein. Agar file ka naam `multi_query_retriever_by_ollama.py` hai:

```powershell
python .\multi_query_retriever_by_ollama.py
```

Ollama mein models available hone chahiye. Is code ko run karne ke liye OpenAI API key required nahi hai.

---

## 18. Complete workflow diagram

```text
User enters a question
          |
          v
Multi-Query Retriever
          |
          v
Qwen2.5:3b generates alternative queries
          |
          v
Original query + alternative queries
          |
          v
Ollama Embeddings (nomic-embed-text)
          |
          v
FAISS searches top matching documents
for each query (k = 3 per query)
          |
          v
Combine retrieved documents
          |
          v
Remove duplicate page_content entries
          |
          v
Print source, topic, content and count
```

---

## 19. Key terms — quick revision

| Term | Easy meaning |
|---|---|
| Embedding | Text ka numerical vector representation |
| Vector store | Embeddings ko store aur search karne wala component |
| FAISS | Similar vectors/documents search karne ki library |
| Retriever | Query ke liye relevant documents dhoondhta hai |
| Multi-Query Retriever | Ek question se multiple search queries use karta hai |
| LLM | Text samajhne aur generate karne wala language model |
| Metadata | Document ke baare mein extra information |
| `include_original=True` | Original question ko bhi retrieval mein include karna |
| `k=3` | Har individual query par top 3 matching documents maangna |
| Deduplication | Repeated results ko remove karna |

---

## 20. Final summary

Is project mein `nomic-embed-text` document embeddings banata hai, FAISS similar documents search karta hai, aur `qwen2.5:3b` original question ke alternate versions generate karta hai. `MultiQueryRetriever` original aur alternative queries ke results combine karta hai. Aakhir mein code same `page_content` wale duplicate documents remove karke source, topic, content aur unique results ki total count print karta hai.

**Yaad rakhein:** Multi-Query Retriever ka main goal ek hi question ko multiple search phrasings se explore karna hai. Isse relevant documents milne ke chances badh sakte hain, lekin zyada documents milna automatically better relevance ki guarantee nahi hai.
