from collections import defaultdict
from typing import List

def reciprocal_rank_fusion(rankings: List[List[int]], k: int=5):
    rrf = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking, 1):
            rrf[doc_id] += 1.0 / (k + i)

    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)