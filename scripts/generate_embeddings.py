import os
import sys
import pickle
import logging

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Ensure app module is importable from root
sys.path.insert(0, os.getcwd())

from app.loader import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = settings['model'].get('name', "sentence-transformers/all-MiniLM-L6-v2")
CSV_PATH = "data/products.csv"
OUTPUT_PICKLE_PATH = "data/emb.pkl"


def main():
    # 1. Validate CSV
    if not os.path.exists(CSV_PATH):
        logger.error(f"{CSV_PATH} not found. Aborting.")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logger.error(f"Failed to read {CSV_PATH}: {e}")
        sys.exit(1)

    df = df.reset_index(drop=True)

    # Validate required columns
    required_cols = ['product_id', 'product_name', 'description']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"CSV is missing required columns: {missing}")
        sys.exit(1)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"\n{df.head()}")

    # 2. Load Model
    logger.info(f"Loading model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load model '{MODEL_NAME}': {e}")
        sys.exit(1)

    # 3. Prepare texts
    logger.info("Preparing texts...")
    texts = (
        df["product_name"].fillna("").astype(str) + " " +
        df["description"].fillna("").astype(str)
    ).tolist()

    # 4. Generate embeddings
    logger.info("Generating embeddings...")
    try:
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)

    logger.info(f"Embedding shape: {embeddings.shape}")

    # 5. Save
    embedding_data = {
        "embeddings": embeddings,
        "product_ids": df["product_id"].tolist(),
        "df_index": df.index.tolist()
    }

    try:
        os.makedirs(os.path.dirname(OUTPUT_PICKLE_PATH), exist_ok=True)
        with open(OUTPUT_PICKLE_PATH, "wb") as f:
            pickle.dump(embedding_data, f)
        logger.info(f"{OUTPUT_PICKLE_PATH} saved successfully!")
    except IOError as e:
        logger.error(f"Failed to save pickle: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
