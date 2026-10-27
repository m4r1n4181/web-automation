import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env at the beginning of config
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories automatically
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Web app settings
WEB_URL = os.getenv("WEB_URL", os.getenv("WEB_URL", ""))
WEB_USERNAME = os.getenv("WEB_USERNAME", os.getenv("USERNAME", ""))
WEB_PASSWORD = os.getenv("WEB_PASSWORD", os.getenv("PASSWORD", ""))

# OCR settings
VISION_AGENT_API_KEY = os.getenv("VISION_AGENT_API_KEY", "")

# Database settings
DB_PATH = DATA_DIR / "payments.db"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = LOGS_DIR / "app.log"

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/EXP2_fuzzy_plus_engineering__RandomForest.pkl")

# PDF Extraction Coordinates
PDF_COORDS = {
    "NAME_MIN_X": 145,
    "NAME_MAX_X": 250,
    "RATA_MIN_X": 470,
    "RATA_MAX_X": 510,
    "ROW_TOL": 2,
    "BRANCH_MIN_X": 100,
    "BRANCH_MAX_X": 400,
    "BRANCH_MIN_Y": 80,
    "BRANCH_MAX_Y": 150,
}

