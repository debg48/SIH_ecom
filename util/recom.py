from sentence_transformers import SentenceTransformer, util
import pickle
import os

class Recommender:
    def __init__(self, model_name, emb_path, prod_path):
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        if not os.path.exists(prod_path):
            raise FileNotFoundError(f"Products file not found: {prod_path}")
            
        self.model = SentenceTransformer(model_name)
        self.emb = pickle.load(open(emb_path, 'rb'))
        self.prod_name = pickle.load(open(prod_path, 'rb'))

    def recommend(self, name):
        q_emb = self.model.encode(name)
        scores = util.cos_sim(q_emb, self.emb)[0].cpu().tolist()
        doc_score_pairs = list(zip(self.prod_name, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        # Return top 5 matches (excluding the exact match if it's the first one, though logic used 1:6)
        return doc_score_pairs[1:6]
