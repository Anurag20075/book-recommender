# 📚 AI-Powered Book Recommendation System

### *Semantic Search + Emotion-Aware Re-ranking*

Traditional book search relies on exact keyword matching, which fails to capture the "vibe" or deep thematic meaning of a story. This system uses **Transformer-based NLP models** to understand descriptions semantically and filter them based on the emotional response they evoke.

---

## 🧠 The AI Stack

The system utilizes three specialized Hugging Face models to process text:

### 1. Semantic Vector Search

* **Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Role:** Converts book descriptions into 384-dimensional dense vectors (embeddings).
* **Function:** It allows the system to calculate **Cosine Similarity** between a user's query and the database. If you search for "lonely traveler in space," it will find books about "solitary galactic voyages" even without keyword overlap.

### 2. Zero-Shot Classification (Thematic Filtering)

* **Model:** `valhalla/distilbart-mnli-12-3`
* **Role:** Categorizes books into dynamic labels (e.g., *Dystopian, Romance, Mystery*) without the model being specifically trained on those categories.
* **Function:** Handles "cold-start" scenarios where metadata is missing, allowing users to filter results by themes on the fly.

### 3. Emotion Classification (Sentiment Search)

* **Model:** `j-hartmann/emotion-english-distilroberta-base`
* **Role:** Analyzes the "emotional fingerprint" of the text across 7 categories: **Joy, Sadness, Anger, Fear, Surprise, Disgust,** and **Neutral**.
* **Function:** This provides the **Sentiment-Aware Filter**. It ensures that if a user wants a "heartwarming" story, the system boosts books with high "Joy" scores and penalizes those with high "Sadness" or "Fear" scores.

---

## 🏗️ System Architecture

1. **Preprocessing:** `Setup_data.py` cleans the `tagged_descriptions.text` and prepares it for embedding.
2. **Vector Store:** Descriptions are encoded using `all-MiniLM-L6-v2`.
3. **Search Phase:**
* Query is embedded into the same vector space.
* Top candidates are retrieved using Cosine Similarity.


4. **Refinement Phase:**
* Candidates are passed through the **Emotion Model** to check for "vibe" alignment.
* **Zero-Shot** classification validates that the book fits the requested genre/theme.


5. **Output:** A ranked list of books that are both semantically and emotionally relevant.

---

## 📂 Project Structure

```bash
book-recommender/
│
├── Main.py                 # Application entry point (CLI/Logic)
├── UI.py                   # Interface and interaction handling
├── Setup_data.py           # Dataset preprocessing & embedding generation
├── Test_API.py             # Validation for HF model endpoints
├── checkmodelfile.py       # Local model integrity verification
│
├── Sentiment_Search.ipynb  # Dev Lab: Emotion & Zero-shot logic
├── vector_Search.ipynb     # Dev Lab: Vector similarity & search
│
├── tagged_descriptions.text # Raw book data
└── requirements.txt        # Model dependencies (Transformers, Torch, etc.)

```

---

## ▶️ Getting Started

1. **Clone the Repository**
```bash
git clone https://github.com/Anurag20075/book-recommender.git
cd book-recommender

```


2. **Install Requirements**
```bash
pip install -r requirements.txt

```


3. **Run Data Setup**
*This will download the Dataset for our ML Project from Kaggle.*
```bash
python Setup_data.py

```


4. **Launch the Recommender**
```bash
python Main.py

```



---

## 👤 Author

**Anurag**
1. *Software Developer | NLP & Backend Enthusiast*
2. *GitHub: [@Anurag20075](https://github.com/Anurag20075)*

---
