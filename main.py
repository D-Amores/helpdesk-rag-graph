import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag.rag_pipeline import RAGPipeline


def main():
    print("🎧 Testing RAG Pipeline")
    print("=" * 40)

    pipeline = RAGPipeline()
    vectorstore = pipeline.build(force_rebuild=True)

    if vectorstore:
        retriever = pipeline.get_retriever(vectorstore)
        results = retriever.search("resetear contraseña")
        print(retriever.format_results(results))


if __name__ == "__main__":
    main()
