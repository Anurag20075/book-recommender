# import os
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # 1. Load the API key from .env
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# if api_key:
#     print("✅ Success: API Key found in .env!")
# else:
#     print("❌ Error: GOOGLE_API_KEY not found. Check your .env file.")

# # 2. Try to initialize Gemini Embeddings
# try:
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     print("✅ Success: Gemini Embeddings initialized!")
# except Exception as e:
#     print(f"❌ Error: Could not initialize Gemini. {e}")



import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Force reload the .env file to ensure we use the NEW key
load_dotenv(override=True)

def verify_new_key():
    print(f"--- Testing New Key: {os.getenv('GOOGLE_API_KEY')[:10]}... ---")
    
    # We use 'gemini-2.0S-flash' because it is the default free model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    try:
        print("Sending request to Google...")
        response = llm.invoke("Reply with the word 'Works'.")
        print(f"✅ SUCCESS! The API replied: {response.content}")
        print("Your 'Limit: 0' error is fixed.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_new_key()