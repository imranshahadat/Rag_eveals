from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langsmith import traceable
from app.logger import error_msg

class LocalDocumentLoader:
    @traceable(name="Local PDF Loader", run_type="tool", tags=["pdf", "local"], metadata={"loader_version": "1.0"})
    def load_pdf(self, file_path: str):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"PDF loaded successfully from {file_path}")
            return documents
        except Exception as e:
            error_msg(f"PDF loading failed: {e}")
            raise

    @traceable(name="Local Text Loader", run_type="tool", tags=["text", "local"], metadata={"loader_version": "1.0"})
    def load_text(self, file_path: str):
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            print(f"Text loaded successfully from {file_path}")
            return documents
        except Exception as e:
            error_msg(f"Text loading failed: {e}")
            raise

    @traceable(name="Local Directory Loader", run_type="tool", tags=["directory", "local"])
    def load_directory(self, folder_path: str, glob_pattern: str = "**/*.pdf"):
        try:
            loader = DirectoryLoader(folder_path, glob=glob_pattern, loader_cls=PyPDFLoader)
            documents = loader.load()
            print(f"Documents loaded successfully from {folder_path}")
            return documents
        except Exception as e:
            error_msg(f"Directory loading failed: {e}")
            raise
