#core/config.py
import os
from dotenv import load_dotenv

# Load .env early so modules reading os.getenv at import-time get expected values.
load_dotenv()

TESTING = os.getenv("TESTING", "false").lower() in ("1", "true", "yes", "on")
