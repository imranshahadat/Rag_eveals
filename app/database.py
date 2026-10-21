import psycopg2
from app.config import settings
from app.logger import log_msg, error_msg

def get_db_connection():
    if not settings.DATABASE_URL:
        error_msg("DATABASE_URL is not set")
        return None
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                user_query TEXT,
                assistant_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cursor.close()
        log_msg("Database connection established and table verified.")
        return conn
    except Exception as e:
        error_msg(f"Database connection failed: {e}")
        return None
