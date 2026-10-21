import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
assert key, "GEMINI_API_KEY not found in .env"
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

client = genai.Client(api_key=key)
r = client.models.generate_content(model=model, contents="Reply with the single word: ready")
print(model, "->", r.text)