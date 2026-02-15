import os
import sys
import argparse
import pickle
import logging

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Ensure app module is importable from root
sys.path.insert(0, os.getcwd())

from app.loader import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Setup argument parser
parser = argparse.ArgumentParser(description="Migrate products to Qdrant")
parser.add_argument('--mode', type=str, required=True, choices=['replace', 'drop', 'append'],
                    help="Migration mode: replace (upsert by ID), drop (recreate collection), append (add new IDs)")
args = parser.parse_args()

# Configuration
QDRANT_URL = settings['qdrant']['url']
QDRANT_API_KEY = settings['qdrant']['api_key']
COLLECTION_NAME = settings['qdrant'].get('collection_name', 'products')
MODEL_NAME = settings['model']['name']
CSV_PATH = "data/products.csv"
PICKLE_PATH = "data/emb.pkl"


def load_embeddings(df):
    """
    Load pre-computed embeddings from pickle if available.
    Falls back to generating them with the configured model.
    """
    if os.path.exists(PICKLE_PATH):
        logger.info(f"Loading pre-computed embeddings from {PICKLE_PATH}...")
        try:
            with open(PICKLE_PATH, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'embeddings' in data:
                embeddings = data['embeddings']
            else:
                embeddings = data

            logger.info(f"Loaded {len(embeddings)} embeddings.")

            if len(embeddings) != len(df):
                logger.warning(
                    f"Embedding count ({len(embeddings)}) != product count ({len(df)}). "
                    "Falling back to regeneration."
                )
                return generate_embeddings(df)

            return embeddings

        except Exception as e:
            logger.warning(f"Failed to load pickle ({e}). Falling back to regeneration.")
            return generate_embeddings(df)
    else:
        logger.info(f"No pre-computed embeddings found at {PICKLE_PATH}.")
        return generate_embeddings(df)


def generate_embeddings(df):
    """Generate embeddings from scratch using the configured model."""
    logger.info(f"Loading model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load model '{MODEL_NAME}': {e}")
        sys.exit(1)

    logger.info("Generating embeddings...")
    sentences = (df['product_name'].fillna('') + " " + df['description'].fillna('')).tolist()

    try:
        embeddings = model.encode(sentences, show_progress_bar=True)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)

    return embeddings


def main():
    logger.info(f"--- Starting Migration in {args.mode.upper()} mode ---")

    # 1. Validate CSV
    if not os.path.exists(CSV_PATH):
        logger.error(f"{CSV_PATH} not found. Aborting.")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logger.error(f"Failed to read {CSV_PATH}: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} rows.")

    # Validate required columns
    required_cols = ['product_id', 'product_name', 'description']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"CSV is missing required columns: {missing}")
        sys.exit(1)

    # 2. Connect to Qdrant
    if not QDRANT_URL or QDRANT_URL.startswith('$'):
        logger.error("QDRANT_URL is not set. Check your .env file.")
        sys.exit(1)

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant at '{QDRANT_URL}': {e}")
        sys.exit(1)

    # 3. Handle 'Drop' Mode
    if args.mode == 'drop':
        try:
            logger.info(f"Dropping collection '{COLLECTION_NAME}'...")
            client.delete_collection(collection_name=COLLECTION_NAME)
            logger.info("Collection dropped.")
        except Exception as e:
            logger.warning(f"Could not drop collection (may not exist): {e}")

    # 4. Ensure Collection Exists
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            vector_size = 384 if 'MiniLM' in MODEL_NAME else 768
            logger.info(f"Creating collection '{COLLECTION_NAME}' (vector_size={vector_size})...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Collection created.")
    except Exception as e:
        logger.error(f"Failed to create/verify collection: {e}")
        sys.exit(1)

    # 5. Determine ID offset (Append mode)
    start_id = 0
    if args.mode == 'append':
        try:
            count = client.count(collection_name=COLLECTION_NAME).count
            start_id = count
            logger.info(f"Appending starts from ID {start_id}.")
        except Exception as e:
            logger.warning(f"Could not fetch count ({e}). Starting from 0.")

    # 6. Load or Generate Embeddings
    embeddings = load_embeddings(df)

    # 7. Build Points and Upload
    logger.info("Uploading to Qdrant...")
    points = []
    for idx, row in df.iterrows():
        payload = row.where(pd.notnull(row), None).to_dict()

        if args.mode == 'append':
            point_id = start_id + idx
        else:
            point_id = idx

        points.append(PointStruct(
            id=point_id,
            vector=embeddings[idx].tolist(),
            payload=payload
        ))

    # Batch Upsert
    BATCH_SIZE = 100
    total = len(points)
    for i in range(0, total, BATCH_SIZE):
        batch = points[i: i + BATCH_SIZE]
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            logger.info(f"Uploaded {min(i + BATCH_SIZE, total)}/{total}")
        except Exception as e:
            logger.error(f"Failed to upload batch starting at {i}: {e}")
            sys.exit(1)

    logger.info(f"Migration completed successfully! {total} points processed.")


if __name__ == "__main__":
    main()
