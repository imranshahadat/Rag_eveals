import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable
from app.document_loader import LocalDocumentLoader
from app.logger import log_msg, error_msg

class DocumentSplitter:
    @traceable(name="Document Splitter", run_type="tool", tags=["splitter"])
    def split_and_index(self, documents, vectorstore, chunk_size=1000, chunk_overlap=200):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(documents)
        print(f"Split documents into {len(all_splits)} chunks.")
        
        if not all_splits:
            print("⚠️ No chunks to index. Check if documents are empty.")
            return 0
            
        vectorstore.add_documents(documents=all_splits)
        print(f"Successfully indexed {len(all_splits)} chunks into vector store.")
        return len(all_splits)

class ResumeAssistant:
    def __init__(self, llm, retriever, memory=None):
        from app.memory import ChatMemoryManager
        self.retriever = retriever
        self.llm = llm
        self.memory = memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question using the following retrieved context. If the context is empty or irrelevant, politely state that you don't have that information in your documents yet.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("user", "{user_message}")
        ])

    def preprocess_message(self, user_message: str) -> str:
        return user_message.strip()

    def _format_docs(self, docs):
        if not docs:
            return "No relevant context found."
        return "\n\n".join(doc.page_content for doc in docs)

    @traceable(name="RAG Pipeline Stream", run_type="chain", tags=["rag", "streaming", "memory"])
    def stream_message(self, user_message: str):
        cleaned_message = self.preprocess_message(user_message)
        if self.memory:
            self.memory.add_user_message(cleaned_message)
        
        context_docs = self.retriever.invoke(cleaned_message)
        context = self._format_docs(context_docs)
        chat_history = self.memory.get_context_messages() if self.memory else []
        
        formatted_prompt = self.prompt.invoke({
            "context": context,
            "chat_history": chat_history,
            "user_message": cleaned_message
        })

        full_response = ""
        for chunk in self.llm.stream(formatted_prompt):
            content = chunk.content if chunk.content else ""
            full_response += content
            yield content
            
        if self.memory:
            self.memory.add_ai_message(full_response)

class IngestionService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.loader = LocalDocumentLoader()
        self.splitter = DocumentSplitter()

    def ingest_single_file(self, file_path: str, file_type: str) -> int:
        if file_type == "pdf":
            documents = self.loader.load_pdf(file_path)
        elif file_type == "text":
            documents = self.loader.load_text(file_path)
        else:
            raise ValueError("Unsupported file type. Use 'pdf' or 'text'.")
        return self.splitter.split_and_index(documents=documents, vectorstore=self.vector_store)

    def ingest_directory(self, folder_path: str) -> int:
        print(f"Scanning directory: {folder_path}")
        documents = self.loader.load_directory(folder_path)
        if not documents:
            print("⚠️ No documents found in the directory.")
            return 0
        return self.splitter.split_and_index(documents=documents, vectorstore=self.vector_store)

class DatabaseService:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def save_chat(self, session_id: str, query: str, response: str):
        if not self.database_url:
            return
        try:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    user_query TEXT,
                    assistant_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
            cursor.execute(
                "INSERT INTO chat_history (session_id, user_query, assistant_response) VALUES (%s, %s, %s);",
                (session_id, query, response)
            )
            conn.commit()
            cursor.close()
            conn.close()
            log_msg(f"Chat saved for session {session_id}")
        except Exception as e:
            error_msg(f"DB save failed: {e}")