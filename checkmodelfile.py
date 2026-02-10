# from google import genai
# import os

# API_KEY = os.getenv("Google_API_KEY")

# try:
#     client = genai.Client(api_key=API_KEY)
    
#     print(f"--- Checking access for Key: {API_KEY[:10]}... ---")
    
#     # List all models available to this key
#     print("\nAvailable Models:")
#     found_any = False
#     for model in client.models.list():
#         # We only care about models that can generate content (not just embeddings)
#         if "generateContent" in model.supported_actions:
#             print(f"✅ {model.name}")
#             found_any = True
            
#     if not found_any:
#         print("❌ No models found! Your key might be invalid or has no permissions.")
        
# except Exception as e:
#     print("\n❌ CRITICAL ERROR:")
#     print(e)
#     print("\nTroubleshooting:")
#     print("1. If you see '403 Permission Denied', your key is invalid.")
#     print("2. If you see '429 Resource Exhausted', you are spamming the check too fast.")
#     print("3. If you see 'Quota limit: 0', you are likely using a Cloud Console key without billing.")




from google import genai

import os
API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

print("--- Embedding Models ---")
for model in client.models.list():
    if "embedContent" in model.supported_actions:
        print(f"✅ {model.name}")
