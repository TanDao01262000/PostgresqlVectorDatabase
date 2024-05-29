
import os
from getpass import getpass
from semantic_router.encoders import HuggingFaceEncoder

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("OpenAI API key: ")

encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")