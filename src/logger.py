import sys
import logging
from src.config import LOG_LEVEL, LOG_FILE

def get_logger(name):
    """
    Returns a logger that writes to both stdout and logs/app.log.
    Format: YYYY-MM-DD HH:MM:SS | LEVEL | module_name | message
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Guard against duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
