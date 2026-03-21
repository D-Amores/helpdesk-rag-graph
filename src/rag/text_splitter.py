from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextSplitter:
    """Handles splitting documents into smaller chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks."""
        print("✂️  Splitting documents into chunks...")

        chunks = self.splitter.split_documents(documents)
        chunks = self._enrich_chunks(chunks)

        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def _enrich_chunks(self, chunks: List[Document]) -> List[Document]:
        """Adds chunk metadata to each chunk."""
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content),
                }
            )
        return chunks
