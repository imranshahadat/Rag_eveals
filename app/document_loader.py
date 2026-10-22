import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langsmith import traceable
from app.logger import error_msg

class LocalDocumentLoader:
    @traceable(name="Local PDF Loader", run_type="tool", tags=["pdf", "local"])
    def load_pdf(self, file_path: str):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"PDF loaded successfully: {file_path} ({len(documents)} pages)")
            return documents
        except Exception as e:
            error_msg(f"PDF loading failed: {e}")
            raise

    @traceable(name="Local Text Loader", run_type="tool", tags=["text", "local"])
    def load_text(self, file_path: str):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            print(f"Text loaded successfully: {file_path}")
            return documents
        except Exception as e:
            error_msg(f"Text loading failed: {e}")
            raise

    @traceable(name="Local Directory Loader", run_type="tool", tags=["directory", "local"])
    def load_directory(self, folder_path: str):
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Directory not found: {folder_path}")
            
            all_documents = []
            # Load PDFs
            pdf_loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
            all_documents.extend(pdf_loader.load())
            
            # Load Text files
            text_loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
            all_documents.extend(text_loader.load())
            
            print(f"Loaded {len(all_documents)} documents from directory: {folder_path}")
            return all_documents
        except Exception as e:
            error_msg(f"Directory loading failed: {e}")
            raise