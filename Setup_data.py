import kagglehub
import pandas as pd
import numpy as np
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
books = pd.read_csv(f"{path}/books.csv")
mask = (
    books["description"].notna() & 
    books["num_pages"].notna() & 
    books["published_year"].notna() & 
    books["average_rating"].notna()
)
books_clean = books[mask].copy()
books_clean["word_count"] = books_clean["description"].str.split().str.len()
books_clean["age_book"] = 2025 - books_clean["published_year"]
books_final = books_clean[books_clean["word_count"] >= 25].copy()
books_final["title_and_subtitle"] = np.where(
    books_final["subtitle"].isna(),
    books_final["title"],
    books_final["title"].astype(str) + ": " + books_final["subtitle"].astype(str)
)
books_final["tagged_description"] = books_final[["isbn13" , "description"]].astype(str).agg(' | '.join, axis=1)
cols_to_drop = ['subtitle', 'description', 'age_book', 'word_count']
books_cleaned_export = books_final.drop(cols_to_drop, axis=1)
books_cleaned_export.to_csv('books_cleaned.csv', index=False)
