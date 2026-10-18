
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document


def main():
    print("\nInitializing local Ollama models...")

    # Local embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Local LLM
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )

    # Sample documents for retrieval
    documents = [
        Document(
            page_content=(
                "Photosynthesis is the process by which green plants use "
                "sunlight, water, and carbon dioxide to produce glucose "
                "and oxygen. Chlorophyll absorbs light energy."
            ),
            metadata={"topic": "photosynthesis"},
        ),
        Document(
            page_content=(
                "Light-dependent reactions occur in the thylakoid "
                "membranes of chloroplasts. They convert light energy "
                "into chemical energy."
            ),
            metadata={"topic": "photosynthesis"},
        ),
        Document(
            page_content=(
                "Plants take in carbon dioxide through tiny openings "
                "in their leaves called stomata. Roots absorb water."
            ),
            metadata={"topic": "photosynthesis"},
        ),
        Document(
            page_content=(
                "The human heart pumps blood throughout the body. "
                "Blood carries oxygen and nutrients to body tissues."
            ),
            metadata={"topic": "human heart"},
        ),
        Document(
            page_content=(
                "Python is a programming language used for automation, "
                "web development, data analysis, and AI applications."
            ),
            metadata={"topic": "Python"},
        ),
    ]

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embeddings)

    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    print("Creating contextual compressor...")
    compressor = LLMChainExtractor.from_llm(llm)

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    print("\n" + "=" * 50)
    print(" CONTEXTUAL COMPRESSION RETRIEVER")
    print(" Powered by local Ollama models")
    print("=" * 50)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Enter your query: ").strip()

            if question.lower() in ("exit", "quit"):
                print("Exiting program. Goodbye!")
                break

            if not question:
                print("Please enter a question.\n")
                continue

            print("\nRetrieving and compressing relevant content...")
            results = retriever.invoke(question)

            print("\n" + "=" * 50)
            print("COMPRESSED RESULTS")
            print("=" * 50)

            if not results:
                print("No relevant results found.")
            else:
                for i, doc in enumerate(results, start=1):
                    print(f"\nResult {i}")
                    print(f"Topic: {doc.metadata.get('topic', 'Unknown')}")
                    print(f"Content: {doc.page_content}")

            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nProgram stopped.")
            break

        except Exception as error:
            print(f"\nError: {error}")
            print("Check whether Ollama is running and the models are installed.\n")


if __name__ == "__main__":
    main()
