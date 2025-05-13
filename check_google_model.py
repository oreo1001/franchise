from google import generativeai as genai
from config import settings
GEMINI_API_KEY = settings.GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

models = genai.list_models()
for m in models:
    for action in m.supported_generation_methods:
        if action == "generateContent":
            print(f"model: {m.name}")
            print(f"supported actions: {m.supported_generation_methods}")
            print(m.name)