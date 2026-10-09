# RAG Basics — Retrieval-Augmented Generation


## 1. LLM and the Need for External Knowledge

### What is an LLM?

LLM stands for **Large Language Model**.

It is a neural-network-based model trained on a very large amount of data.

Generally:

```text
More parameters
      ↓
More powerful model
```

However, an LLM may not give a useful answer when it does not have the required information.

### Examples of problems

1. The model may not know about a user's private/project data.
2. The model may not know about very recent information.
3. The model may confidently provide wrong information — this is related to **hallucination**.
4. A model may not have access to a particular organization's documents or knowledge base.

A simple example:

```text
User: What is inside my company's latest internal document?

LLM:
I don't have access to that private document.
```

---

# 2. Fine-Tuning

Fine-tuning means taking a **pre-trained model** and training it further on a smaller, domain-specific dataset.

Example:

```text
Pre-trained LLM
      ↓
Domain-specific dataset
      ↓
Fine-tuning
      ↓
Domain-specific model
```

For example:

```text
Student
   ↓
LLM
   ↓
Engineering-related training data
   ↓
Fine-tuning
   ↓
Engineering/domain-specific model
```

## Types / approaches mentioned in the notes

### Supervised Fine-Tuning

Provide labeled examples:

```text
Prompt → Desired Output
```

The model learns to produce the desired output.

### Continued Pre-training

Continue training the model on additional domain-related data.

```text
Pre-trained model
      ↓
Additional training
      ↓
Domain knowledge
```

### RLHF

RLHF stands for **Reinforcement Learning from Human Feedback**.

Human feedback is used to improve the model's behavior.

---

# 3. Steps to Carry Out Supervised Fine-Tuning

### Step 1 — Collect Data

Collect a carefully curated dataset.

The notes mention:

- a few hundred examples
- a few thousand examples
- prompt → desired output pairs

The quality of the examples is important.

### Step 2 — Choose a Method

Possible approaches include:

- Full-parameter fine-tuning
- LoRA
- QLoRA
- Parameter-efficient adapters

### Step 3 — Train for a Few Epochs

Keep some model weights frozen or partially frozen and update only the required parameters.

For example:

```text
LoRA:
Only a small subset/adapters are trained.

Full fine-tuning:
All model weights may be trained.
```

### Step 4 — Evaluate and Safety Test

Evaluate the model for:

- factuality
- exact-match rate
- hallucination rate
- safety

Use a held-out dataset for evaluation.

---

# 4. Why We Do Not Use Fine-Tuning Everywhere

Fine-tuning is not always the best solution.

### Reason 1 — Large Model Training Is Expensive

Training large models requires significant computational resources.

```text
Large model
    ↓
More computation
    ↓
More expensive training
```

### Reason 2 — Technical Expertise

Fine-tuning can require technical knowledge and appropriate training infrastructure.

### The alternative

Instead of changing the model's weights, we can provide relevant information to the model at query time.

This leads to **in-context learning** and, for external knowledge retrieval, **RAG**.

---

# 5. In-Context Learning

In-context learning is a capability of LLMs where the model learns how to perform a task from examples placed in the prompt, without updating its model weights.

Examples of models mentioned in the notes include:

- GPT-3.5
- Claude

The model can learn a task by seeing examples in the prompt.

Example:

```text
Text: I love this phone. It is so smooth.
Label: Positive

Text: They are costly a lot.
Label: Negative

Text: The camera is amazing!
Label: Positive

Text: I hate the battery life.
Label: ?
```

The model can infer the expected task from the examples.

---

# 6. Emergent Property

An **emergent property** is a behavior or ability that suddenly appears in a system when it reaches a certain scale or complexity, even though that behavior was not explicitly programmed or expected from individual components.

In LLMs, some capabilities become more noticeable as the model becomes larger and more capable.

---

# 7. What Is RAG?

RAG stands for:

> **Retrieval-Augmented Generation**

RAG is a technique that combines:

- information retrieval
- language generation

The model retrieves relevant information from an external knowledge base and uses that information as context to generate a grounded response.

Simple idea:

```text
External Knowledge
       ↓
    Retrieve
       ↓
Relevant Context
       ↓
      LLM
       ↓
    Response
```

Another simple definition:

> RAG makes a language model smarter for a particular question by giving it relevant external information at the time of the question.

---

# 8. Why Did We Need RAG?

The main motivation is that an LLM's internal knowledge is not always enough.

RAG helps when we need:

### 1. Private Data

Example:

```text
Company documents
Internal PDFs
Project documentation
Private database
```

The LLM can use retrieved private information as context.

### 2. Recent Developments

An LLM's training data may not contain the latest information.

RAG can retrieve updated information from an external source.

### 3. Hallucination Reduction

If the model is instructed to answer from retrieved context, irrelevant guessing can be reduced.

For example:

```text
Context:
[relevant retrieved document]

Question:
[user question]

Instruction:
Answer only from the provided context.
If the context is insufficient, say "I don't know."
```

### 4. Large Models and Fine-Tuning Cost

Instead of retraining model weights whenever knowledge changes, we can update the external knowledge source.

This can be cheaper and easier for many knowledge-based applications.

---

# 9. RAG vs Fine-Tuning

These two approaches solve different problems.

| Feature | RAG | Fine-Tuning |
|---|---|---|
| Main idea | Retrieve external information at query time | Train the model further |
| Changes model weights? | No | Yes |
| Best for | External/current/private knowledge | Behavior, style, task/domain adaptation |
| Updating knowledge | Update knowledge base | Often requires additional training |
| Training required | Usually no model fine-tuning | Yes |
| Uses retrieval | Yes | No |
| Can provide current documents | Yes | Not automatically |
| Cost of updating knowledge | Usually lower | Can be higher |
| Example | Company-document Q&A | Training a model for a specific response style/task |

### Easy way to remember

```text
RAG:
"Don't change the brain.
Give the brain the right information."

Fine-Tuning:
"Train the brain further."
```

### When to choose RAG

Use RAG when:

- information changes frequently
- information is private
- you have a large document collection
- answers should be grounded in documents
- you need source/document-based answers

### When to choose Fine-Tuning

Fine-tuning can be useful when:

- you need a particular behavior or style
- you need task-specific adaptation
- you have suitable training data
- changing the model's learned behavior is the goal

### Can RAG and Fine-Tuning be combined?

Yes.

A system can use a fine-tuned model **and** retrieval.

```text
Fine-tuned LLM
      +
Retrieved Context
      ↓
Final Answer
```

---

# 10. RAG Pipeline

A basic RAG system can be divided into four major stages:

1. Indexing
2. Retrieval
3. Augmentation
4. Generation

```text
Documents
    ↓
Document Loader
    ↓
Text Splitting
    ↓
Embedding Generation
    ↓
Vector Store
    ↓
Retriever
    ↓
Relevant Chunks
    ↓
Augmentation
    ↓
LLM
    ↓
Response
```

---

# 11. Indexing

Indexing means preparing the source knowledge so that it can be searched efficiently.

The notes describe indexing as a process of preparing the knowledge base so that it can be efficiently searched at query time.

The process consists of four major steps:

```text
1. Document Ingestion
2. Text Chunking
3. Embedding Generation
4. Storage in Vector Store
```

---

# 12. Document Ingestion / Document Loader

Document ingestion means loading source knowledge into the RAG system.

Examples mentioned in the notes:

- PDF reports
- Word documents
- YouTube transcripts
- GitHub repositories
- Internet / web pages

The general flow is:

```text
Source
  ↓
Document Loader
  ↓
Document
  ↓
Text
```

### Example

```text
Web page
   ↓
Web/Document Loader
   ↓
Document
   ↓
Text
```

The document loader converts information from a source into a form that can be processed by the RAG pipeline.

---

# 13. Why Document Loading Matters

Different sources have different formats.

A RAG system may need to work with:

```text
PDF
DOCX
TXT
Web pages
YouTube transcripts
GitHub files
```

A loader provides a common way to bring this information into the pipeline.

---

# 14. Text Chunking / Text Splitting

A large document is broken into smaller, meaningful chunks.

Definition:

> Text chunking means breaking a large document into small, semantically meaningful pieces.

### Why do we need chunks?

LLMs have context limits.

For example:

```text
Large Document
      ↓
Text Splitter
      ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

Smaller chunks can also provide more focused semantic search.

### Basic idea

```text
Large document
      ↓
┌───────────────┐
│ Chunk 1       │
├───────────────┤
│ Chunk 2       │
├───────────────┤
│ Chunk 3       │
├───────────────┤
│ Chunk 4       │
└───────────────┘
```

---

# 15. Text Splitter Tools

Tools mentioned in the notes include:

- Recursive Character Text Splitter
- Markdown Header Text Splitter
- Semantic Chunking

Different splitters can be selected depending on the structure of the data.

---

# 16. Embedding Generation

After creating chunks, each chunk is converted into a dense numerical vector called an **embedding**.

Definition:

> An embedding is a dense vector representation that captures the meaning of text.

### Why embeddings?

#### 1. Similar ideas are placed close together in vector space

```text
"car"
"automobile"
"vehicle"

       ↓

Similar vector representations
```

#### 2. Embeddings allow fuzzy / semantic search

The system does not have to depend only on exact keyword matching.

Example:

```text
Question:
"How can I fix my vehicle?"

Document:
"Steps to repair an automobile..."
```

Semantic similarity can connect these ideas.

---

# 17. Embedding Tools

Tools mentioned in the notes include:

- OpenAI embeddings
- Sentence Transformers
- Instructor Embeddings

General flow:

```text
Text Chunk
    ↓
Embedding Model
    ↓
Dense Vector
```

---

# 18. Vector Store

A vector store stores vectors along with the original chunk text and metadata.

```text
Chunk
  +
Metadata
  ↓
Embedding
  ↓
Vector Store
```

The vector store becomes the searchable representation of the external knowledge base.

---

# 19. Vector Database Options

The notes mention:

### Local

- FAISS
- Chroma

### Cloud

- Pinecone
- Weaviate
- Milvus
- Qdrant

The exact choice depends on the project requirements.

---

# 20. Complete Indexing Flow

```text
Source / Web / File
        ↓
Document Loader
        ↓
Large Document
        ↓
Text Splitter
        ↓
Small Chunks
        ↓
Embedding Model
        ↓
Dense Vectors
        ↓
Vector Store
        ↓
External Knowledge Base
```

---

# 21. Retrieval

Retrieval is the real-time process of finding the most relevant pieces of information from the pre-built index based on the user's question.

Simple meaning:

> "From all the knowledge I have, which 3–5 chunks are most useful for answering this query?"

Example:

```text
User Query
    ↓
Retriever
    ↓
Vector Store
    ↓
Semantic Search
    ↓
Most Relevant Chunks
```

---

# 22. Retrieval Steps

### Step 1 — Query → Embedding

Convert the user's question into a vector.

```text
Question
   ↓
Embedding Model
   ↓
Query Vector
```

### Step 2 — Semantic Search

Search the vector store for vectors similar to the query vector.

### Step 3 — Ranking

Rank candidate chunks by relevance.

### Step 4 — Fetch Top Results

Return the most relevant chunks.

For example:

```text
Top 5 relevant chunks
```

---

# 23. Retrieval Example

Suppose the knowledge base contains 1,000 document chunks.

User asks:

```text
"What is the company's leave policy?"
```

The retriever does not send all 1,000 chunks to the LLM.

Instead:

```text
1000 chunks
     ↓
Retriever
     ↓
Semantic Search
     ↓
Rank by relevance
     ↓
Top relevant chunks
     ↓
LLM
```

This makes the context more focused.

---

# 24. Augmentation

Augmentation means combining the retrieved documents/chunks with the user's query to create a new, enriched prompt for the LLM.

```text
Relevant Chunks
      +
User Query
      ↓
Augmented Prompt
      ↓
LLM
```

Example:

```text
Context:
{retrieved relevant chunks}

Question:
{user question}

Instruction:
You are a helpful assistant.
Answer the question only from the provided context.
If the context is insufficient, say you don't know.
```

The retrieved chunks become additional context for the LLM.

---

# 25. Generation

Generation is the final step.

The LLM receives:

- the prompt
- the user's question
- the retrieved context

and generates the answer.

```text
Prompt
  ↓
LLM
  ↓
Response
```

The notes describe generation as the final step where the large language model uses the query and retrieved/augmented context to generate a response.

---

# 26. Complete RAG Pipeline

```text
                    ┌─────────────────┐
                    │  External Data  │
                    │ PDF / Web / DB  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Document Loader │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Text Splitter  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Embedding Model │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Vector Store  │
                    └────────┬────────┘
                             │
                             │
User Query ────────────────► Retriever
                             │
                             ▼
                    Relevant Chunks
                             │
                             ├──────────────┐
                             │              │
                             ▼              ▼
                           Query +      Context
                             │              │
                             └──────┬───────┘
                                    ▼
                                Prompt
                                    │
                                    ▼
                                   LLM
                                    │
                                    ▼
                                 Response
```

---

# 27. RAG Architecture in Simple Words

The complete flow can be remembered as:

```text
INGEST
   ↓
SPLIT
   ↓
EMBED
   ↓
STORE
   ↓
RETRIEVE
   ↓
AUGMENT
   ↓
GENERATE
```

Or:

```text
Documents
   ↓
Loader
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Store
   ↓
Retriever
   ↓
Context
   ↓
LLM
   ↓
Answer
```

---

# 28. Problems Solved by RAG

The notes highlight three important problems:

### 1. Private Data

```text
Private data
     ↓
External knowledge base
     ↓
Retriever
     ↓
LLM
```

### 2. Recent Developments

RAG can use a knowledge source containing recent/current information.

```text
Recent information
       ↓
Knowledge base
       ↓
Retrieval
       ↓
LLM
```

### 3. Hallucination

Retrieved context can ground the answer in available information.

RAG does not guarantee that hallucinations disappear, but grounding the model with relevant context can help reduce unsupported answers.

---

# 29. RAG and Chatbots

A chatbot can use RAG to answer questions using an external knowledge base.

Example:

```text
User
 ↓
Query
 ↓
Retriever
 ↓
External Knowledge Base
 ↓
Relevant Context
 ↓
LLM
 ↓
Chatbot Response
```

This is useful for:

- current information
- private information
- document-based question answering

---

# 30. Benefits of Using RAG

The notes mention:

1. Use of up-to-date information
2. Better privacy
3. No strict limitation based only on the model's document knowledge

RAG can also allow the external knowledge source to be updated without retraining the entire LLM.

---

# 31. One-Line Definitions for Quick Revision

| Term | Simple Meaning |
|---|---|
| LLM | Large Language Model |
| RAG | Retrieval-Augmented Generation |
| Document Loader | Loads external data into the RAG pipeline |
| Text Splitter | Breaks large documents into smaller chunks |
| Chunk | A smaller piece of a document |
| Embedding | Numerical vector representation of text meaning |
| Vector Store | Stores and searches vector representations |
| Retriever | Finds relevant chunks for a query |
| Augmentation | Adds retrieved context to the user's query |
| Generation | LLM creates the final answer |
| Fine-Tuning | Further training of a pre-trained model |
| In-Context Learning | Learning the task from examples in the prompt |
| Hallucination | Unsupported or incorrect information generated by a model |

---

# 32. RAG vs Fine-Tuning — Final Revision

```text
                 RAG
                  │
       ┌──────────┴──────────┐
       │                     │
 External Knowledge      Model weights
       │                     │
   Retrieved at          Changed through
   query time             training
       │                     │
       ▼                     ▼
     RAG                 Fine-Tuning
```

### Remember:

> **RAG changes the information available to the model.**

> **Fine-tuning changes the model's learned behavior/weights.**

For knowledge that changes frequently or belongs to an external/private document collection, RAG is often the more practical starting point.

---

# 33. Learning Roadmap for This Repository

This repository follows the same RAG pipeline step by step:

```text
01. RAG Basics
       ↓
02. Document Loader
       ↓
03. Text Splitter
       ↓
04. Embeddings
       ↓
05. Vector Store
       ↓
06. Retrievers
       ↓
07. Complete RAG Pipeline
       ↓
08. RAG Application / Chatbot
```

The four core components can be remembered as:

```text
Document Loader
      ↓
Text Splitter
      ↓
Vector Store
      ↓
Retriever
```

with **Embeddings** connecting text chunks to the vector store and **LLM generation** happening after retrieval.
