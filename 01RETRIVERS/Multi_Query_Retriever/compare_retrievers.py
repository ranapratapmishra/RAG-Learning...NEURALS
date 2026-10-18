
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ----------------------------------------
# 1. LOCAL MODELS
# ----------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# ----------------------------------------
# 2. SAME DOCUMENTS FOR BOTH RETRIEVERS
# ----------------------------------------

documents = [
    Document(
        page_content=(
            "Regular exercise, walking, and proper sleep "
            "can help improve daily energy levels."
        ),
        metadata={"source": "H1", "topic": "health"}
    ),
    Document(
        page_content=(
            "A balanced diet containing fruits, vegetables, "
            "protein, and whole grains supports overall wellness."
        ),
        metadata={"source": "H2", "topic": "nutrition"}
    ),
    Document(
        page_content=(
            "Drinking enough water throughout the day "
            "helps maintain hydration and normal body functions."
        ),
        metadata={"source": "H3", "topic": "hydration"}
    ),
    Document(
        page_content=(
            "A consistent sleep schedule and a relaxing "
            "bedtime routine can support better sleep quality."
        ),
        metadata={"source": "H4", "topic": "sleep"}
    ),
    Document(
        page_content=(
            "Mindfulness, deep breathing, and short breaks "
            "may help people manage everyday stress."
        ),
        metadata={"source": "H5", "topic": "stress"}
    ),
    Document(
        page_content=(
            "Plants use sunlight, water, and carbon dioxide "
            "to produce sugars during photosynthesis."
        ),
        metadata={"source": "I1", "topic": "science"}
    ),
    Document(
        page_content=(
            "Python is a programming language often used "
            "for automation, data analysis, and AI projects."
        ),
        metadata={"source": "I2", "topic": "programming"}
    ),
]


# ----------------------------------------
# 3. HELPER: REMOVE DUPLICATE DOCUMENTS
# ----------------------------------------

def remove_duplicates(docs):
    unique_docs = {}

    for doc in docs:
        unique_docs[doc.page_content] = doc

    return list(unique_docs.values())


# ----------------------------------------
# 4. NORMAL RETRIEVER
# ----------------------------------------

def run_normal_retriever(retriever, question):
    print("\n" + "=" * 55)
    print("NORMAL RETRIEVER")
    print("=" * 55)

    results = retriever.invoke(question)

    results = remove_duplicates(results)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print("Topic:", doc.metadata.get("topic", "Unknown"))
        print("Content:", doc.page_content)

    print("\nNormal Retriever total:", len(results))

    return results


# ----------------------------------------
# 5. MULTI-QUERY RETRIEVER
# ----------------------------------------

def run_multi_query_retriever(retriever, question):
    print("\n" + "=" * 55)
    print("MULTI-QUERY RETRIEVER")
    print("=" * 55)

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

    if isinstance(response_text, list):
        response_text = " ".join(
            str(part) for part in response_text
        )

    # Parse generated queries
    generated_queries = []

    for line in str(response_text).splitlines():
        query = line.strip()

        # Remove numbering and common bullet symbols
        query = query.lstrip("0123456789.):-* ").strip()

        if query and query.lower() not in [
            q.lower() for q in generated_queries
        ]:
            generated_queries.append(query)

    # Always include the original question
    queries = [question]

    for query in generated_queries:
        if query.lower() != question.lower():
            queries.append(query)

    print("\nQueries used for retrieval:")

    for i, query in enumerate(queries, start=1):
        print(f"{i}. {query}")

    # Search with every query
    all_results = []

    for query in queries:
        query_results = retriever.invoke(query)
        all_results.extend(query_results)

    # Combine and deduplicate retrieved documents
    results = remove_duplicates(all_results)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print("Topic:", doc.metadata.get("topic", "Unknown"))
        print("Content:", doc.page_content)

    print("\nMulti-Query Retriever total:", len(results))

    return results, queries


# ----------------------------------------
# 6. COMPARISON
# ----------------------------------------

def main():
    print("Creating FAISS vector store using local embeddings...")

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Both methods use the same base retriever
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    print("\nLocal models and vector store are ready.")

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        normal_results = run_normal_retriever(
            base_retriever,
            question
        )

        multi_results, queries = run_multi_query_retriever(
            base_retriever,
            question
        )

        normal_sources = {
            doc.page_content for doc in normal_results
        }

        multi_sources = {
            doc.page_content for doc in multi_results
        }

        common_docs = normal_sources & multi_sources
        extra_docs = multi_sources - normal_sources

        print("\n" + "=" * 55)
        print("FINAL COMPARISON")
        print("=" * 55)

        print("Original question:", question)
        print("Normal Retriever documents:", len(normal_results))
        print("Multi-Query documents:", len(multi_results))
        print("Documents found by both:", len(common_docs))
        print("Additional documents from Multi-Query:", len(extra_docs))
        print("Alternative queries generated:", len(queries) - 1)

        print("\nNote:")
        print(
            "More documents do not automatically mean better results. "
            "Check their relevance to the question."
        )

    except Exception as error:
        print("\nAn error occurred:", error)
        print(
            "Check that Ollama is running and both models "
            "are downloaded."
        )


if __name__ == "__main__":
    main()
