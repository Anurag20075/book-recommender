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
