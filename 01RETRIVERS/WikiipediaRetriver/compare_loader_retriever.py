from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import WikipediaRetriever


def main():
    print("=" * 60)
    print("PART 1: DOCUMENT LOADER")
    print("=" * 60)

    # Load the complete local text file
    loader = TextLoader("cricket.txt", encoding="utf-8")
    local_docs = loader.load()

    print(f"Documents loaded: {len(local_docs)}")
    print("\nLoaded content:")
    print(local_docs[0].page_content)

    print("\nLoader metadata:")
    print(local_docs[0].metadata)

    print("\n" + "=" * 60)
    print("PART 2: WIKIPEDIA RETRIEVER")
    print("=" * 60)

    # Create the Wikipedia Retriever
    retriever = WikipediaRetriever(
        top_k_results=2,
        lang="en"
    )

    query = input(
        "\nEnter a query for Wikipedia: "
    ).strip()

    if not query:
        print("Please enter a valid query.")
        return

    try:
        # Search Wikipedia using the query
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            print("\nNo Wikipedia documents found.")
            return

        print(
            f"\nDocuments retrieved: {len(retrieved_docs)}"
        )

        for index, doc in enumerate(
            retrieved_docs, start=1
        ):
            print("\n" + "-" * 60)
            print(f"RETRIEVED DOCUMENT {index}")
            print("-" * 60)

            print("\nContent:")
            print(doc.page_content[:500])

            print("\nMetadata:")
            print(doc.metadata)

    except Exception as error:
        print("\nWikipedia retrieval failed:", error)
        print("Check your internet connection and try again.")


if __name__ == "__main__":
    main()