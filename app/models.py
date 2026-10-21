from pydantic import BaseModel, Field

class IngestRequest(BaseModel):
    file_path: str
    file_type: str = Field(default="pdf", pattern="^(pdf|text)$")
