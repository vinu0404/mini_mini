import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set GITHUB_TOKEN environment variable")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Directory Configuration
PERSISTENT_DIR = Path("gitspeak_persistent")
PERSISTENT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DB = str(PERSISTENT_DIR / "checkpoints.sqlite")

# Global State Management
GLOBAL_RETRIEVERS = {}
GLOBAL_SESSIONS = {}
GLOBAL_FOLDER_STRUCTURES = {}

# LLM Temperature Settings
BASIC_TEMP = 0.1
CODE_ANALYST_TEMP = 0.1
SECURITY_TEMP = 0.2
PERFORMANCE_TEMP = 0.2
QUALITY_TEMP = 0.3
AGENT_SELECTION_TEMP = 0.1
PRESENT_TELLER_TEMP = 0.1

# Token Limits
MAX_TOKENS_STANDARD = 6000
MAX_TOKENS_QUALITY = 10000
MAX_TOKENS_AGENT_SELECTION = 2000
MAX_TOKENS_PRESENT_TELLER = 500

# Processing Limits
MAX_FILES_TO_PROCESS = 1500
MAX_FOLDER_DEPTH = 8
MAX_OPEN_PRS = 50
MAX_CONVERSATION_HISTORY_CHARS = 2500

# Chart Configuration
DEFAULT_CHART_WIDTH = 800
DEFAULT_CHART_HEIGHT = 600
QUALITY_CHART_WIDTH = 900
QUALITY_CHART_HEIGHT = 700

# Diagram Configuration
DEFAULT_DIAGRAM_WIDTH = 1000
DEFAULT_DIAGRAM_HEIGHT = 800
MERMAID_MAX_URL_SAFE_LENGTH = 32

# File Extensions for Analysis
ANALYZABLE_EXTENSIONS = (
    ".py", ".ipynb", ".c", ".cpp", ".js", ".ts", ".tsx", ".jsx", ".go", 
    ".java", ".rb", ".php", ".swift", ".sh", ".bash", ".ps1", ".r", ".R", 
    ".yaml", ".yml", ".json", ".xml", ".html", ".htm", ".css", ".scss", 
    ".sass", ".sql", ".dockerfile", ".Dockerfile", ".md", ".markdown", 
    ".makefile", ".Makefile", ".mk"
)

# Important Files
IMPORTANT_FILES = [
    'readme.md', 'readme.txt', 'readme.rst', 'main.py', 'app.py', 
    'index.js', 'package.json', 'requirements.txt', 'setup.py', 
    'dockerfile', 'makefile'
]

# Directories to Skip
SKIP_DIRECTORIES = ['__pycache__', 'node_modules', '.git', 'venv', 'env']

# API URLs
QUICKCHART_BASE_URL = "https://quickchart.io/chart"
MERMAID_BASE_URL = "https://mermaid.ink/img/"
GITHUB_API_BASE_URL = "https://api.github.com"

# Timeout Settings
HTTP_TIMEOUT = 30
USER_INPUT_TIMEOUT = 300

# Logging Configuration
import logging
logging.basicConfig(level=logging.INFO)