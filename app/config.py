import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    # LangSmith
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-eveals-project")

    # LLM (Novita)
    NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "apodex/apodex-1.1-mini")
    NOVITA_BASE_URL = os.getenv("NOVITA_BASE_URL", "https://api.novita.ai/openai")

    # Embeddings (Vercel AI Gateway)
    VERCEL_AI_GATEWAY_API_KEY = os.getenv("VERCEL_AI_GATEWAY_API_KEY")
    VERCEL_AI_GATEWAY_BASE_URL = os.getenv("VERCEL_AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    VERCEL_EMBEDDING_MODEL = os.getenv("VERCEL_EMBEDDING_MODEL", "cohere/embed-v5.0-fast")

    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-eveals-index-v2")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "documents")
    PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "2048")) # Updated to 2048
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")

settings = Settings()