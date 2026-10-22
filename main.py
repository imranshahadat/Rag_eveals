import os
from app.config import settings

# Set LangSmith env vars before importing app modules
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

from app.embedding import embed_model
from app.vectorstore import pinecone_manager
from app.llm import model
from app.memory import SessionManager
from app.services import ResumeAssistant, IngestionService, DatabaseService

print("Initializing RAG Assistant...")

# Initialize Vector Store
vector_store = pinecone_manager.get_vector_store(embedding_model=embed_model.embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize Services
session_manager = SessionManager(llm=model.llm, max_tokens=1200, summary_trigger=10, keep_last_n=4)
assistant = ResumeAssistant(llm=model.llm, retriever=retriever)
ingestion_service = IngestionService(vector_store=vector_store)
db_service = DatabaseService(database_url=settings.DATABASE_URL)

print("Initialization complete!\n")

if __name__ == "__main__":
    session_id = "default_session"
    print("--- RAG Assistant CLI ---")
    print("Commands:")
    print(" /ingest <file_path> <pdf|text>  - Ingest a single local document")
    print(" /ingest_dir <folder_path>       - Ingest ALL PDF/TXT files in a directory")
    print(" /clear                          - Clear conversation memory")
    print(" /exit                           - Quit the application")
    print("-----------------------------\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Session ended. Goodbye!")
                break

            if user_input.startswith("/ingest_dir "):
                folder_path = user_input.split(" ", 1)[1].strip()
                print(f"Ingesting directory: {folder_path}...")
                try:
                    chunks = ingestion_service.ingest_directory(folder_path)
                    print(f"Successfully indexed {chunks} chunks from directory.")
                except Exception as e:
                    print(f"Directory ingestion failed: {e}")
                continue

            if user_input.startswith("/ingest "):
                parts = user_input.split()
                if len(parts) >= 3:
                    file_path = parts[1]
                    file_type = parts[2].lower()
                    if file_type in ["pdf", "text"]:
                        print(f"Ingesting {file_path}...")
                        try:
                            chunks = ingestion_service.ingest_single_file(file_path, file_type)
                            print(f"Successfully indexed {chunks} chunks.")
                        except Exception as e:
                            print(f"Ingestion failed: {e}")
                    else:
                        print("Invalid file type. Use 'pdf' or 'text'.")
                else:
                    print("Usage: /ingest <file_path> <pdf|text>")
                continue

            if user_input.lower() == "/clear":
                session_manager.clear_session(session_id)
                assistant.memory = session_manager.get_memory(session_id)
                print("Conversation memory cleared.")
                continue

            # Normal chat interaction
            print("Assistant: ", end="", flush=True)
            assistant.memory = session_manager.get_memory(session_id)
            full_response = ""
            try:
                for chunk in assistant.stream_message(user_input):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print() # Newline
                
                # Save to DB
                db_service.save_chat(session_id, user_input, full_response)
            except Exception as e:
                print(f"\nError during generation: {e}")

        except KeyboardInterrupt:
            print("\nSession ended. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")