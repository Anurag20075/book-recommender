# import os
# import time
# import pandas as pd
# import typing_extensions as typing
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from google.api_core import exceptions as google_exceptions

# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# # 1. Load your API key
# load_dotenv()

# # 2. This is the code you asked about (The "Chat/Brain" part)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-lite", 
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# # 3. You ALSO need this for the Vector Database (The "Embedding" part)
# # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")
# try:
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     # Test it with a tiny query
#     embeddings.embed_query("test")
#     print("✅ Using text-embedding-004")
# except Exception:
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     print("⚠️ Fallback: Using embedding-001")

# import pandas as pd 

# books=pd.read_csv("books_cleaned.csv")

# books["tagged_description"].to_csv("tagged_descriptions.text",
#                                     sep="\n" , index=False, header=False)


# # 1. Ensure TextLoader is imported (Fixes the NameError from before)
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter

# raw_data = TextLoader("tagged_descriptions.text").load()

# # 2. Set chunk_size to 1000 (roughly the length of a book summary)
# # and chunk_overlap to 100 (keeps context between chunks)
# text_splitter = CharacterTextSplitter(
#     chunk_size=1000, 
#     chunk_overlap=100, 
#     separator="\n"
# )

# # 3. This will now run without the ValueError
# documents = text_splitter.split_documents(raw_data)

# import time
# import typing_extensions as typing
# from google.api_core import exceptions as google_exceptions

# # 1. Define the Data Structure
# class BookAnalysis(typing.TypedDict):
#     broad_category: str
#     emotional_tone: str

# # 2. Define the Helper Function (The Brain)
# def analyze_book_safe(tagged_descriptions):
#     # Skip bad data
#     if not tagged_descriptions or len(str(tagged_descriptions)) < 20:
#         return "Other", "Neutral"

#     prompt = f"""
#     You are a librarian. Analyze this book description:
#     "{tagged_descriptions}"
    
#     Task 1: Classify it into ONE broad category: [Fiction, Nonfiction].
#     Task 2: Identify the dominant emotional tone: [Happy, Sad, Suspenseful, Inspiring, Dark, Educational].
    
#     Return the result in JSON.
#     """
    
#     # We use "with_structured_output" to guarantee clean data
#     # Note: Ensure 'llm' is defined in a previous cell!
#     structured_llm = llm.with_structured_output(BookAnalysis)
    
#     try:
#         result = structured_llm.invoke(prompt)
#         return result['broad_category'], result['emotional_tone']
#     except Exception as e:
#         return "Other", "Neutral"

# # 3. Define the Main Loop (The Safe Processor)
# def process_books_safely(df):
#     mask = (df['broad_category'] == "Other") | (df['broad_category'].isna())
#     to_process = df[mask]
    
#     print(f"📚 Starting Safe Processing for {len(to_process)} books...")
    
#     counter = 0
#     for index, row in to_process.iterrows():
#         try:
#             category, emotion = analyze_book_safe(row['description'])
            
#             df.at[index, 'broad_category'] = category
#             df.at[index, 'emotional_tone'] = emotion 
            
#             print(f"[{counter+1}/{len(to_process)}] {row['title'][:20]}... -> {category} | {emotion}")
            
#             time.sleep(4) # Safety Brake
#             counter += 1
            
#             if counter % 20 == 0:
#                 df.to_csv("books_in_progress.csv", index=False)
#                 print("💾 Progress Saved.")
                
#         except google_exceptions.ResourceExhausted:
#             print("🛑 Quota Hit! Sleeping for 60 seconds...")
#             time.sleep(60)
#         except Exception as e:
#             print(f"⚠️ Unexpected Error: {e}")
#             time.sleep(5)

#     df.to_csv("books_completed.csv", index=False)
#     print("✅ All Done!")

#     category_mapping = {
#     'Fiction': "Fiction",
#     'Juvenile Fiction': "Children's Fiction",
#     'Biography & Autobiography': "Nonfiction",
#     'History': "Nonfiction",
#     'Literary Criticism': "Nonfiction",
#     'Philosophy': "Nonfiction",
#     'Religion': "Nonfiction",
#     'Comics & Graphic Novels': "Fiction",
#     'Drama': "Fiction",
#     'Juvenile Nonfiction': "Children's Nonfiction",
#     'Science': "Nonfiction",
#     'Poetry': "Fiction"
# }
# # Apply Mapping (Fast Way)
# books['broad_category'] = books['categories'].map(category_mapping)

# # Cleanup for the "Smart Way"
# books['broad_category'] = books['broad_category'].fillna("Other")
# if 'emotional_tone' not in books.columns:
#     books['emotional_tone'] = "Neutral"

# process_books_safely(books)

# documents = []
# for index, row in books.iterrows():
#     cat = row['broad_category'] if pd.notna(row['broad_category']) else "Other"
#     emo = row['emotional_tone'] if pd.notna(row['emotional_tone']) else "Neutral"

#     doc = Document(
#         page_content=str(row['tagged_description']), 
#         metadata={
#             "title": row['title'],
#             "authors": row['authors'],
#             "broad_category": cat,
#             "emotional_tone": emo 
#         }
#     )
#     documents.append(doc)

# print(f"✅ Created {len(documents)} documents ready for the DB.")


# import chromadb
# from langchain_chroma import Chroma
# import time
# # --- PART A: INITIALIZE ---
# # We use the raw chromadb client to "touch" the file first
# # client = chromadb.PersistentClient(path=db_path)
# db_path = "final_books_db" 
# print("✅ Native client connected successfully.")

# # --- PART B: START THE LANGCHAIN DB ---
# # Now we pass that existing client to LangChain
# db_books = Chroma.from_documents(
#     documents=documents[:40], 
#     embedding=embeddings, 
#     persist_directory=db_path
# )

# # --- PART C: BATCH LOOP ---
# batch_size = 40
# for i in range(batch_size, len(documents), batch_size):
#     batch = documents[i : i + batch_size]
#     db_books.add_documents(batch)
#     print(f"✅ Stored up to {i + len(batch)} books.")
#     time.sleep(10)


#     import os
# from langchain_chroma import Chroma

# # Use the exact same path and embedding function
# db_path = "final_books_db"

# if os.path.exists(db_path):
#     # This only LOADS the database. It does NOT create it or call the API for new embeddings.
#     db_books = Chroma(
#         persist_directory=db_path,
#         embedding_function=embeddings
#     )
#     print(f"✅ Database loaded successfully with {len(db_books.get()['ids'])} books.")
# else:
#     print("❌ Error: Database folder not found. Did you delete it?")