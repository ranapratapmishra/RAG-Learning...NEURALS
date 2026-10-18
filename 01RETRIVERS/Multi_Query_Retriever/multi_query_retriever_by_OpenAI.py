"""
Multi-Query Retriever demo using LangChain, FAISS, and local Hugging Face embeddings.

Note:
- Query expansion uses an OpenAI chat model, so OPENAI_API_KEY is required.
- Embeddings are generated locally with sentence-transformers.
"""

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


def main():
    # 1. Prepare sample documents. Metadata helps identify each document.
    all_docs = [
        Document(
            page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
            metadata={"source": "H1"},
        ),
        Document(
            page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
            metadata={"source": "H2"},
        ),
        Document(
            page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
            metadata={"source": "H3"},
        ),
        Document(
            page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
            metadata={"source": "H4"},
        ),
        Document(
            page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
            metadata={"source": "H5"},
        ),
        Document(
            page_content="The solar energy system in modern homes helps balance electricity demand.",
            metadata={"source": "I1"},
        ),
        Document(
            page_content="Python balances readability with power, making it a popular system design language.",
            metadata={"source": "I2"},
        ),
        Document(
            page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
            metadata={"source": "I3"},
        ),
        Document(
            page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
            metadata={"source": "I4"},
        ),
        Document(
            page_content="Black holes bend spacetime and store immense gravitational energy.",
            metadata={"source": "I5"},
        ),
    ]

    try:
        # 2. Create local embeddings. The first run downloads the model.
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 3. Build a FAISS index from the documents and embeddings.
        vectorstore = FAISS.from_documents(
            documents=all_docs,
            embedding=embedding_model,
        )

        # 4. Build a regular similarity retriever for comparison.
        similarity_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # 5. MultiQueryRetriever uses an LLM to create alternative queries.
        # This part calls the OpenAI API and needs OPENAI_API_KEY.
        query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        multiquery_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            llm=query_llm,
        )

        # 6. Ask the user for a query.
        query = input("Enter your question: ").strip()
        if not query:
            print("Please enter a question.")
            return

        # 7. Run both retrievers.
        similarity_results = similarity_retriever.invoke(query)
        multiquery_results = multiquery_retriever.invoke(query)

        # 8. Print results for a clear comparison.
        print("\n" + "=" * 60)
        print("REGULAR SIMILARITY SEARCH")
        print("=" * 60)
        if not similarity_results:
            print("No documents found.")
        for i, doc in enumerate(similarity_results, start=1):
            print(f"\n--- Result {i} | Source: {doc.metadata.get('source', 'unknown')} ---")
            print(doc.page_content)

        print("\n" + "=" * 60)
        print("MULTI-QUERY RETRIEVER RESULTS")
        print("=" * 60)
        if not multiquery_results:
            print("No documents found.")
        for i, doc in enumerate(multiquery_results, start=1):
            print(f"\n--- Result {i} | Source: {doc.metadata.get('source', 'unknown')} ---")
            print(doc.page_content)

    except Exception as error:
        print("\nAn error occurred:", error)
        print(
            "Check installed packages, internet access for the model download, "
            "and OPENAI_API_KEY for the Multi-Query LLM."
        )


if __name__ == "__main__":
    main()
