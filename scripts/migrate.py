import os
import sys
import argparse
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Ensure app module is importable from root
import sys
sys.path.append(os.getcwd())

from app.loader import settings

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

def main():
    print(f"--- Starting Migration in {args.mode.upper()} mode ---")
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # 1. Load Data
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    # 2. Check/Initialize Qdrant Connection
    if not QDRANT_URL:
        print("Error: QDRANT_URL is not set in environment or config.")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 3. Handle 'Drop' Mode
    if args.mode == 'drop':
        print(f"Dropping collection '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("Collection dropped.")
    
    # 4. Ensure Collection Exists (for all modes)
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        # Determine vector size based on model (384 for MiniLM-L6, 768 for mpnet)
        vector_size = 384 if 'MiniLM' in MODEL_NAME else 768
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Collection created with vector size {vector_size}.")

    # 5. Determine ID Start (for Append mode)
    start_id = 0
    if args.mode == 'append':
        # Fetch current count / max ID logic
        # For simplicity, we can inspect collection info or just count
        # A robust append needs to know the LAST inserted ID.
        print("Fetching current collection info to determine offset...")
        try:
            count = client.count(collection_name=COLLECTION_NAME).count
            start_id = count
            print(f"Appending starts from ID {start_id}.")
        except Exception as e:
            print(f"Error fetching count: {e}. Starting from 0.")
            start_id = 0

    # 6. Load Embeddings
    PICKLE_PATH = "data/emb.pkl"
    if os.path.exists(PICKLE_PATH):
        print(f"Loading pre-computed embeddings from {PICKLE_PATH}...")
        import pickle
        with open(PICKLE_PATH, 'rb') as f:
            data = pickle.load(f)
            # data is expected to be a dict with key 'embeddings' or a numpy array/list directly
            if isinstance(data, dict) and 'embeddings' in data:
                embeddings = data['embeddings']
            else:
                embeddings = data
        print(f"Loaded {len(embeddings)} embeddings.")
        
        if len(embeddings) != len(df):
            print(f"Warning: Number of embeddings ({len(embeddings)}) does not match number of products ({len(df)}).")
            # In a real scenario, you might want to stop or regenerate.
            # For now, let's assume they are aligned by index as per generate_embeddings.py
    else:
        print(f"Pre-computed embeddings not found at {PICKLE_PATH}.")
        print(f"Loading model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        
        print("Generating embeddings...")
        # Combine name and description
        sentences = (df['product_name'].fillna('') + " " + df['description'].fillna('')).tolist()
        embeddings = model.encode(sentences, show_progress_bar=True)

    # 7. Upload to Qdrant
    print("Uploading to Qdrant...")
    points = []
    for idx, row in df.iterrows():
        # Clean payload (NaN values break JSON serialization)
        payload = row.where(pd.notnull(row), None).to_dict()
        
        # Determine Point ID
        if args.mode == 'append':
            point_id = start_id + idx
        else:
            # For replace/drop, use the DataFrame index (or a 'product_id' column if integer)
            # Assuming product_id in CSV is string "IND-...", we use integer index for Qdrant Point ID
            point_id = idx 

        points.append(PointStruct(
            id=point_id,
            vector=embeddings[idx].tolist(),
            payload=payload
        ))

    # Batch Upsert
    BATCH_SIZE = 100
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"Uploaded batch {i} - {i + len(batch)}")

    print(f"Migration completed successfully! Total {len(points)} points processed.")

if __name__ == "__main__":
    # Ensure app module is findable if running from root
    import sys
    sys.path.append(os.getcwd())
    main()
