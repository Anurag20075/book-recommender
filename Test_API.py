import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Load the API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print("✅ Success: API Key found in .env!")
else:
    print("❌ Error: GOOGLE_API_KEY not found. Check your .env file.")

# 2. Try to initialize Gemini Embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("✅ Success: Gemini Embeddings initialized!")
except Exception as e:
    print(f"❌ Error: Could not initialize Gemini. {e}")