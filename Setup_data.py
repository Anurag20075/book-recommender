import kagglehub
import pandas as pd
import numpy as np

#  Download and Load Dataset
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
books = pd.read_csv(f"{path}/books.csv")

#  Define the Cleaning Mask
mask = (
    books["description"].notna() & 
    books["num_pages"].notna() & 
    books["published_year"].notna() & 
    books["average_rating"].notna()
)

#  Apply Mask and Create a Clean Copy
books_clean = books[mask].copy()

#  Feature Engineering: Word Count and Age
books_clean["word_count"] = books_clean["description"].str.split().str.len()
books_clean["age_book"] = 2025 - books_clean["published_year"]

#  Filter for Quality Descriptions (>= 25 words)
books_final = books_clean[books_clean["word_count"] >= 25].copy()

# Combine Title and Subtitle
books_final["title_and_subtitle"] = np.where(
    books_final["subtitle"].isna(),
    books_final["title"],
    books_final["title"].astype(str) + ": " + books_final["subtitle"].astype(str)
)

# Tag description with ISBN13 for unique identification in NLP tasks
books_final["tagged_description"] = books_final[["isbn13" , "description"]].astype(str).agg(' | '.join, axis=1)

# 7. Final Cleanup and Export
# Fixed 'age_of_book' to 'age_book' to match your calculation
cols_to_drop = ['subtitle', 'description', 'age_book', 'word_count']
books_cleaned_export = books_final.drop(cols_to_drop, axis=1)
books_cleaned_export.to_csv('books_cleaned.csv', index=False)

# --- Terminal Output ---
print(f"Success! Final dataset saved with {len(books_final)} books.")
print("\nTop 5 Categories in Cleaned Data:")
print(books_final["categories"].value_counts().head(5))

print("\nSample of Processed Features:")
print(books_final[['title_and_subtitle', 'tagged_description']].head())




# 