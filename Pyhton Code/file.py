import os
import shutil
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})

db_path = "./book_db_final_version" 
if os.path.exists(db_path):
    shutil.rmtree(db_path)

batch_size = 40
db_books = Chroma.from_documents(
    documents=documents[:batch_size], 
    embedding=hf_embeddings, 
    persist_directory=db_path
)

for i in range(batch_size, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    db_books.add_documents(batch)
    time.sleep(5)