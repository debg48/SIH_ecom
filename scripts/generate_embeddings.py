import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys

# Ensure app module is importable from root
sys.path.append(os.getcwd())

from app.loader import settings

# Configuration
# Using model name from config if available, otherwise fallback to the user's provided string
MODEL_NAME = settings['model'].get('name', "sentence-transformers/all-MiniLM-L6-v2")
CSV_PATH = "data/products.csv"
OUTPUT_PICKLE_PATH = "data/emb.pkl"

def main():
    print(f"Loading data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)

    # Ensure strict sequential indexing
    df = df.reset_index(drop=True)

    print("Dataset shape:", df.shape)
    print(df.head())

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Preparing texts...")
    texts = (
        df["product_name"].fillna("").astype(str) + " " +
        df["description"].fillna("").astype(str)
    ).tolist()

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Embedding shape:", embeddings.shape)

    embedding_data = {
        "embeddings": embeddings,              # shape (N, 384)
        "product_ids": df["product_id"].tolist(),
        "df_index": df.index.tolist()
    }

    print(f"Saving to {OUTPUT_PICKLE_PATH}...")
    with open(OUTPUT_PICKLE_PATH, "wb") as f:
        pickle.dump(embedding_data, f)

    print(f"{OUTPUT_PICKLE_PATH} saved successfully!")

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    main()
