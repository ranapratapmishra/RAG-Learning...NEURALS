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