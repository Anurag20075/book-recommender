import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr

load_dotenv()

# --- 1. DATA AND EMBEDDING SETUP ---
books = pd.read_csv("books_with_emotions.csv")

# Fix: Mapping 'thumbnail' to our logic and handling high-res formatting
books["thumbnail"] = books["thumbnail"].fillna("Cover not available")
books["thumbnail"] = np.where(
    books["thumbnail"].str.contains("googleusercontent|books.google", na=False),
    books["thumbnail"] + "&fife=w800",
    books["thumbnail"]
)

# Initialize HuggingFace Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
db_path = "./book_db_final_version"

# --- 2. LOAD OR CREATE VECTOR DB ---
if os.path.exists(db_path):
    print("✅ Loading existing Vector DB...")
    db_books = Chroma(
        persist_directory=db_path,
        embedding_function=hf_embeddings
    )
else:
    print("🚧 Vector DB not found. Building it now...")
    raw_documents = TextLoader("tagged_descriptions.txt").load()
    # Note: CharacterTextSplitter with 0 chunk_size will treat each line as a document
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    
    db_books = Chroma.from_documents(
        documents, 
        hf_embeddings, 
        persist_directory=db_path
    )
    print("✅ Vector DB built and saved!")

# --- 3. RECOMMENDATION LOGIC ---
def retrive_semantic_recommendations(query: str, category: str = None, tone: str = None, intial_top_k: int = 30, final_top_k: int = 10):
    recs = db_books.similarity_search(query, k=intial_top_k)
    
    # Extract ISBNs (Assuming your tagged_descriptions.txt starts each line with the isbn13)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].copy()
    
    if category != "All":
        book_recs = book_recs[book_recs["books_category"] == category]
    
    tone_map = {
        "Happy": "joy",
        "Sad": "sadness",
        "Angry": "anger",
        "Surprised": "surprise",
        "Suspensful": "fear"
    }
    
    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)
    
    return book_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrive_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        # Using 'tagged_description' from your column list for the snippet
        desc = row["tagged_description"] if pd.notna(row["tagged_description"]) else ""
        truncated_desc = " ".join(str(desc).split()[:25]) + "..."
        
        # Author formatting logic
        authors_raw = str(row["authors"])
        authors_split = authors_raw.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        caption = f"**{row['title']}**\nBy {authors_str}\n\n{truncated_desc}"
        # Use the correct 'thumbnail' column here
        results.append((row["thumbnail"], caption))
        
    return results

# --- 4. GRADIO INTERFACE ---
categories = ["All"] + sorted(books["books_category"].dropna().unique().tolist())
tones = ["ALL", "Happy", "Sad", "Angry", "Surprised", "Suspensful"]

with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommendation System")
    gr.Markdown("Find books based on meaning, category, and emotional resonance.")
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Search Query", 
                placeholder="e.g., A dark mystery set in a small snowy town"
            )
            category_input = gr.Dropdown(label="Category", choices=categories, value="All")
            tone_input = gr.Dropdown(label="Emotional Tone", choices=tones, value="ALL")
            recommend_button = gr.Button("Find My Next Read", variant="primary")
        
        with gr.Column(scale=2):
            # Using columns=2 for better visibility of covers
            output_gallery = gr.Gallery(label="Recommended Books", columns=2, height="auto", object_fit="contain")

    recommend_button.click(
        fn=recommend_books, 
        inputs=[query_input, category_input, tone_input], 
        outputs=output_gallery
    )

if __name__ == "__main__":
    dashboard.launch()