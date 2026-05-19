from bm25 import BM25
from rrf import reciprocal_rank_fusion

def dense_vector_search(query: str, k: int=5):
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]

def hybrid_search(query, k=20):
    _, dense_search_ranking = dense_vector_search(query, 100)
    _, bm25_search_ranking  = BM25.get_top_k(query, 100)

    results = reciprocal_rank_fusion([dense_search_ranking, bm25_search_ranking], k=k)
    return results
