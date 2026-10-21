import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "health-assistant-project")
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    MODEL_NAME = "apodex/apodex-1.1-mini"
    NOVITA_BASE_URL = "https://api.novita.ai/openai"
    
    PINECONE_INDEX_NAME = "colabragindex"
    PINECONE_NAMESPACE = "pdf-documents"
    PINECONE_DIMENSION = 256

settings = Settings()
