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
# Load data and ensure ISBNs are strings for consistent matching
books = pd.read_csv("books_with_emotions.csv")
books["isbn13"] = books["isbn13"].astype(str)

# Robust Thumbnail Fix
def fix_thumbnail(url):
    if pd.isna(url) or str(url).lower() == "cover not available":
        return "https://via.placeholder.com/400x600?text=No+Cover+Available"
    
    # Ensure high-res for Google images
    if any(domain in str(url) for domain in ["googleusercontent", "books.google"]):
        return str(url).split('&fife=')[0] + "&fife=w800"
    return url

books["thumbnail"] = books["thumbnail"].apply(fix_thumbnail)

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
    # Split by line (one book per line)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    
    db_books = Chroma.from_documents(
        documents, 
        hf_embeddings, 
        persist_directory=db_path
    )
    print("✅ Vector DB built and saved!")

# --- 3. RECOMMENDATION LOGIC ---
def retrieve_semantic_recommendations(query: str, category: str = "All", tone: str = "ALL", initial_top_k: int = 30, final_top_k: int = 12):
    # Search vector DB
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Extract ISBNs as strings to match the dataframe
    # Assuming tagged_descriptions.txt format: "9781234567890 Description text..."
    books_list = [rec.page_content.strip('"').split()[0] for rec in recs]
    
    # Filter the main dataframe
    book_recs = books[books["isbn13"].isin(books_list)].copy()
    
    # Filter by Category
    if category != "All":
        book_recs = book_recs[book_recs["books_category"] == category]
    
    # Sort by Emotion
    tone_map = {
        "Happy": "joy",
        "Sad": "sadness",
        "Angry": "anger",
        "Surprised": "surprise",
        "Suspenseful": "fear"
    }
    
    if tone in tone_map:
        emotion_col = tone_map[tone]
        if emotion_col in book_recs.columns:
            book_recs = book_recs.sort_values(by=emotion_col, ascending=False)
    
    return book_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return []
        
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        # Title & Author Formatting
        title = row.get("title", "Unknown Title")
        authors_raw = str(row.get("authors", "Unknown Author"))
        authors_split = authors_raw.split(";")
        
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        # Description Snippet
        desc = row.get("tagged_description", "")
        truncated_desc = " ".join(str(desc).split()[:20]) + "..."
        
        caption = f"**{title}**\nBy {authors_str}\n\n{truncated_desc}"
        results.append((row["thumbnail"], caption))
        
    return results

# --- 4. GRADIO INTERFACE ---
categories = ["All"] + sorted(books["books_category"].dropna().unique().tolist())
tones = ["ALL", "Happy", "Sad", "Angry", "Surprised", "Suspenseful"]

with gr.Blocks(theme=gr.themes.Soft(), css="#header {text-align: center; margin-bottom: 20px;}") as dashboard:

    with gr.Column(elem_id="header"):
        gr.Markdown("# 📚 Semantic Book Finder")
        gr.Markdown("Find books by **vibes**, not just keywords.")

    with gr.Tabs():
        with gr.Tab("🔍 Discover"):
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1):
                    query_input = gr.Textbox(
                        label="Describe what you're looking for",
                        placeholder="A story about a lonely robot in space..."
                    )
                    category_input = gr.Dropdown(
                        label="Category",
                        choices=categories,
                        value="All"
                    )
                    tone_input = gr.Dropdown(
                        label="Emotional Tone",
                        choices=tones,
                        value="ALL"
                    )
                    with gr.Row():
                        recommend_button = gr.Button("🔍 Find Books", variant="primary")
                        clear_button = gr.Button("Clear")

                # Results
                with gr.Column(scale=2):
                    output_gallery = gr.Gallery(
                        label="Recommendations",
                        columns=3,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                        allow_preview=True
                    )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### System Architecture
            1. **Semantic Search**: Uses `all-MiniLM-L6-v2` to understand the meaning of your query.
            2. **Vector Storage**: Powered by `ChromaDB`.
            3. **Emotion Ranking**: Re-ranks results based on emotional scores (Joy, Fear, Sadness, etc.) pre-calculated in the dataset.
            """)

    # Actions
    recommend_button.click(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=[output_gallery]
    )

    clear_button.click(
        fn=lambda: ("", "All", "ALL", None),
        inputs=[],
        outputs=[query_input, category_input, tone_input, output_gallery]
    )

if __name__ == "__main__":
    dashboard.launch()