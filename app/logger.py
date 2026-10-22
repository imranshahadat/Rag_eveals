import logging
import logging.handlers
from pathlib import Path

# 1. Setup Log Directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# 2. Create Logger
logger = logging.getLogger("health_assistant")
logger.setLevel(logging.DEBUG) # Capture everything (DEBUG level), handlers will filter

# 3. Define a highly informative format
log_format = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 4. Console Handler (StreamHandler) - Shows INFO and above in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)

# 5. Rotating File Handler - MANAGES FILE SIZE AUTOMATICALLY
# maxBytes=5MB: Rolls over to a new file when it hits 5 Megabytes
# backupCount=3: Keeps only the 3 most recent log files (app.log, app.log.1, app.log.2, app.log.3)
# This prevents your disk from filling up if the app runs for months or processes huge files.
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, 
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3, 
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG) # Save everything (including DEBUG) to the file
file_handler.setFormatter(log_format)

# 6. Add handlers to logger
# Clear existing handlers to prevent duplicate logs if this module is reloaded
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# --- Helper Functions (Backward Compatible) ---

def log_msg(message: str, **kwargs):
    """Log standard info. Use kwargs to add extra context (e.g., extra={'file': 'resume.pdf'})"""
    logger.info(message, **kwargs)

def warning_msg(message: str, **kwargs):
    """Log warnings."""
    logger.warning(message, **kwargs)

def error_msg(message: str, exc: Exception = None):
    """
    Log an error. 
    If an exception object is passed, it will automatically capture and log the full traceback.
    """
    if exc:
        logger.error(f"{message} | Exception Details: {str(exc)}", exc_info=exc)
    else:
        logger.error(message)

def log_exception(message: str):
    """
    It automatically captures the full stack trace of the current exception.
    """
    logger.exception(message)

# Log startup to confirm rotation is active
logger.info("Logging system initialized. RotatingFileHandler active (Max 5MB, 3 backups).")