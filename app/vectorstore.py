import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langsmith import traceable
from app.config import settings
from app.logger import error_msg

class PineconeManager:
    def __init__(self, index_name: str = settings.PINECONE_INDEX_NAME, namespace: str = settings.PINECONE_NAMESPACE):
        self.index_name = index_name
        self.namespace = namespace
        self.pc = None
        self.vector_store = None

    def initialize_api_key(self):
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing.")
        os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
        print("Pinecone API key loaded")

    @traceable(name="PineCone init", run_type="tool", tags=["setup_index", "pinecone"], metadata={"Pinecone_version": "1.0"})
    def setup_index(self, dimension: int = settings.PINECONE_DIMENSION):
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            existing_indexes = [index["name"] for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"Pinecone index '{self.index_name}' created")
            else:
                print(f"Pinecone index '{self.index_name}' already exists")
        except Exception as e:
            print(f"Pinecone index error: {e}")
            raise

    @traceable(name="Pinecone VectorStore Init", run_type="tool", tags=["vectorstore", "pinecone"])
    def get_vector_store(self, embedding_model):
        try:
            self.vector_store = PineconeVectorStore(
                index=self.pc.Index(self.index_name),
                embedding=embedding_model,
                namespace=self.namespace
            )
            print("Pinecone vector store initialized")
            return self.vector_store
        except Exception as e:
            print(f"Vector store initialization failed: {e}")
            raise

pinecone_manager = PineconeManager()
pinecone_manager.initialize_api_key()
pinecone_manager.setup_index()
