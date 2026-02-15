import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class Recommender:
    def __init__(self, model_name, qdrant_url, qdrant_api_key, collection_name):
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        if not qdrant_url:
            raise ValueError("qdrant_url cannot be empty. Check your .env file.")

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}': {e}")

        try:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant at '{qdrant_url}': {e}")

        self.collection_name = collection_name

        # Verify collection exists
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.warning(f"Collection '{self.collection_name}' does not exist in Qdrant. "
                               "Run the migration script first.")
        except Exception as e:
            logger.warning(f"Could not verify collection existence: {e}")

    def recommend(self, query):
        """
        Generate recommendations for the given query string.
        Returns a list of (product_name, score) tuples.
        """
        if not query or not query.strip():
            return []

        try:
            query_vector = self.model.encode(query).tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to encode query '{query}': {e}")

        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=5
            )
        except UnexpectedResponse as e:
            raise RuntimeError(f"Qdrant search failed (collection may not exist): {e}")
        except Exception as e:
            raise ConnectionError(f"Failed to communicate with Qdrant: {e}")

        results = []
        for point in search_result.points:
            product_name = point.payload.get('product_name', 'Unknown Product')
            results.append((product_name, round(point.score, 4)))

        return results
