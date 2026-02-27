#core/config.py
import os

TESTING = os.getenv("TESTING", "false").lower() == "true"