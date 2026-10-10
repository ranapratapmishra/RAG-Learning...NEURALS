# from langchain_community.retrievers import WikipediaRetriever


# def main():
#     # Step 1: Create the Wikipedia Retriever
#     retriever = WikipediaRetriever(
#         top_k_results=2,
#         lang="en"
#     )

#     # Step 2: Enter your search query
#     query = input("Enter your question: ")

#     # Step 3: Retrieve relevant documents
#     try:
#         docs = retriever.invoke(query)

#         # Step 4: Display the results
#         if not docs:
#             print("\nNo documents found.")
#             return

#         print(f"\nTotal documents retrieved: {len(docs)}")

#         for index, doc in enumerate(docs, start=1):
#             print("\n" + "=" * 60)
#             print(f"DOCUMENT {index}")
#             print("=" * 60)

#             print("\nContent:")
#             print(doc.page_content[:1000])

#             print("\nMetadata:")
#             print(doc.metadata)

#     except Exception as error:
#         print("\nAn error occurred:", error)
#         print("Check your internet connection and installed packages.")


# if __name__ == "__main__":
#     main()

try:
    from langchain_community.retrievers import WikipediaRetriever
except ImportError:
    print("Required packages are not installed.")
    print("Run: python -m pip install -r requirements.txt")
    raise SystemExit(1)


def main():
    # Step 1: Create the Wikipedia Retriever
    retriever = WikipediaRetriever(
        top_k_results=2,
        lang="en"
    )

    # Step 2: Enter your search query
    query = input("Enter your question: ").strip()

    if not query:
        print("Please enter a valid question.")
        return

    # Step 3: Retrieve relevant documents
    try:
        docs = retriever.invoke(query)

        # Step 4: Display the results
        if not docs:
            print("\nNo documents found.")
            return

        print(f"\nTotal documents retrieved: {len(docs)}")

        for index, doc in enumerate(docs, start=1):
            print("\n" + "=" * 60)
            print(f"DOCUMENT {index}")
            print("=" * 60)

            print("\nContent:")
            print(doc.page_content[:1000])

            print("\nMetadata:")
            print(doc.metadata)

    except Exception as error:
        print("\nAn error occurred:", error)
        print("Check your internet connection and try again.")


if __name__ == "__main__":
    main()