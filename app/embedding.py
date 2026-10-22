import os
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

class VercelEmbeddings:
    @traceable(
        name="Vercel Embed Model",
        run_type="tool",
        tags=["model", "vercel-ai-gateway"],
        metadata={"provider": "vercel"}
    )
    def __init__(
        self, 
        model_name: str = None,
        api_key: str = None,
        base_url: str = None
    ):
        self.model_name = model_name or os.getenv("VERCEL_EMBEDDING_MODEL")
        self.api_key = api_key or os.getenv("VERCEL_AI_GATEWAY_API_KEY")
        self.base_url = base_url or os.getenv("VERCEL_AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")

        if not self.api_key:
            raise ValueError("VERCEL_AI_GATEWAY_API_KEY is missing.")

        try:
            self.embedding_model = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                # Add this to ensure it handles single strings correctly
                check_embedding_ctx_length=False, 
            )
            print(f"Vercel AI Gateway embedding model ({self.model_name}) initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Vercel embedding model: {e}")
            raise

embed_model = VercelEmbeddings()