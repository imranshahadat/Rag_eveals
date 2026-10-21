from langchain_openai import ChatOpenAI
from langsmith import traceable
from app.config import settings
from app.logger import log_msg, error_msg

class OpenAIModel:
    @traceable(name="OpenAIModel", run_type="tool", tags=["Model_name", settings.MODEL_NAME])
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.8,
                max_tokens=12000
            )
            log_msg(f"Novita LLM ({model_name}) initialized successfully")
            print(f"Novita LLM ({model_name}) initialized")
        except Exception as e:
            error_msg(f"Novita initialization failed: {e}")
            raise

model = OpenAIModel(
    model_name=settings.MODEL_NAME,
    api_key=settings.NOVITA_API_KEY,
    base_url=settings.NOVITA_BASE_URL,
)
