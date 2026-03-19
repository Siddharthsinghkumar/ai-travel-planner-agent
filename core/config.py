#core/config.py
from dotenv import load_dotenv
from core.env_config import get_env_bool

# Load .env early so modules reading os.getenv at import-time get expected values.
load_dotenv()

TESTING = get_env_bool("TESTING", default=False)
