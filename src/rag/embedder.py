from langchain_openai import OpenAIEmbeddings
from config import EMBEDDINGS_MODEL, OPENAI_API_KEY


class EmbeddingProvider:
    """Handles ONLY the connection to OpenAI to generate embeddings."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY
        )

    def get_embeddings(self):
        """Returns the embeddings object ready to use."""
        return self.embeddings
