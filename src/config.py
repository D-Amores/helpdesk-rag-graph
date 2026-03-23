import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Models
EMBEDDINGS_MODEL = "text-embedding-3-small"

# Paths
CHROMADB_PATH = "./chroma_db"
DOCS_PATH = "./docs"
