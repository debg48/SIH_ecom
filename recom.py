from sentence_transformers import SentenceTransformer, util
import pickle

def recommend(name):
    q_emb = model.encode(name)
    scores = util.cos_sim(q_emb, emb)[0].cpu().tolist()
    doc_score_pairs = list(zip(prod_name, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x:x[1], reverse=True)
    return doc_score_pairs[1:6]

emb = pickle.load(open('emb.pkl','rb'))
prod_name = pickle.load(open('prod.pkl','rb'))
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
