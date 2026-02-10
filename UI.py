# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# from langchain_community.document_loaders import TextLoader, DataFrameLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
# import gradio as gr

# load_dotenv()

# books = pd.read_csv("books_with_emotions.csv")
# books["large_thumbnail"] = books["large_thumbnail"] + "&fife=w800"
# books["large_thumbnail"]= np.where(
#             books["large_thumbnail"].isna(),
#             "Cover not available",
#             books["large_thumbnail"],
#             )
# raw_documents = TextLoader("tagged_descriptions.txt").load()
# text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
# db_book= Chroma.from_documents( documents, GoogleGenerativeAIEmbeddings())

# def retrive_semantic_recommendations(query: str,
#                                      category: str =  None,
#                                      tone : str = None,
#                                      intial_top_k : int = 30,
#                                         final_top_k : int = 10

                                     
#                 ) -> pd.DataFrame:
#     recs= db_book.similarity_search(query, k=intial_top_k)
#     books_list=[int (rec.page_content.strip('"').split()[0]) for rec in recs]
#     book_recs=books[books["isbn13"].isin(books_list)].head(intial_top_k)
    
#     if category != "All":
#         book_recs=book_recs[book_recs["books_category"]==category].head(final_top_k)
#     else:
#         book_recs=book_recs.head(final_top_k)

#     if tone == "Happy":
#         book_recs.sort_values(by="joy", ascending=False, inplace=True)
#     elif tone == "Sad":
#         book_recs.sort_values(by="sadness", ascending=False, inplace=True)
#     elif tone == "Angry":
#         book_recs.sort_values(by="anger", ascending=False, inplace=True)
#     elif tone == "Surprised":
#         book_recs.sort_values(by="surprise", ascending=False, inplace=True)
#     elif tone == "Suspensful":
#         book_recs.sort_values(by="fear", ascending=False, inplace=True)

#     return book_recs 

# def recommend_books(query: str, category: str, tone: str) -> pd.DataFrame:
#     recommendations = retrive_semantic_recommendations(query, category, tone)
#     rerult =  []
#     for index, row in recommendations.iterrows():
#         description = row["description"]
#         turncated_desc_split=description.split()
#         turncated_description = " ".join(turncated_desc_split[:20]) + "..." 
        
#         authors_split= row["authors"].split(";")
#         if  len(authors_split) ==  2:
#             authors_str = authors_split[0] + " and " + authors_split[1]
#         elif len(authors_split) > 2:
#             authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
#         else:
#             authors_str = row["authors"]

#         caption = f"{row['title']} by {authors_str}:{turncated_description}"
#         rerult.append((caption, row["large_thumbnail"],caption))
#     return rerult

# categories = ["All"] + sorted(books["books_category"].unique())
# tones=["ALL"] + ["Happy", "Sad", "Angry", "Surprised", "Suspensful"]

# with gr.Blocks(theme =gr.themes.Glass()) as dashboard:
#     gr.Markdown("## Semantic Book Recommendation System")

#     with gr.Row():
#         with gr.Column():
#             query_input = gr.Textbox(label="Enter your query", placeholder="What kind of book are you looking for?")
#             category_input = gr.Dropdown(label="Select Category", choices=categories, value="All")
#             tone_input = gr.Dropdown(label="Select Tone", choices=tones, value="ALL")
#             recommend_button = gr.Button("Recommend Books")
#         with gr.Column():
#             gr.Markdown("### Recommended Books will appear here:")
#             output_gallery = gr.Gallery(label="Recommended Books").style(grid=[1], height="auto")

#     recommend_button.click(fn=recommend_books, inputs=[query_input, category_input, tone_input], outputs=output_gallery)

# if __name__ == "__main__":
#     dashboard.launch()

import pandas as pd
import numpy as np
# Load the CSV
books = pd.read_csv("books_with_emotions.csv")

# --- DEBUG: Run this to see what columns actually exist ---
print("Your CSV columns are:", books.columns.tolist())

# Check for common naming variations and rename to match your code
rename_dict = {
    "thumbnail": "large_thumbnail",
    "image": "large_thumbnail",
    "Large_Thumbnail": "large_thumbnail"
}
books.rename(columns=rename_dict, inplace=True)

# Safety: Check if it exists now, if not, create a dummy column so the code doesn't crash
if "large_thumbnail" not in books.columns:
    print("⚠️ Column 'large_thumbnail' not found. Creating a placeholder.")
    books["large_thumbnail"] = "Cover not available"
else:
    # Now run your original logic
    books["large_thumbnail"] = books["large_thumbnail"].fillna("Cover not available")
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].str.contains("http", na=False),
        books["large_thumbnail"] + "&fife=w800",
        books["large_thumbnail"]
    )