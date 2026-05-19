import math
import numpy as np
from typing import List
from transformers import PreTrainedTokenizer
from collections import defaultdict

class BM25:
    def __init__(self, corpus: List[List[str]], tokenizer:PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.corpus    = corpus
        self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
        self.n_docs       = len(self.tokenized_corpus)
        self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / self.n_docs

        self.idf          = self._calculate_idf()
        self.term_freqs   = self._calculate_term_freqs()
        
    def _calculate_idf(self):
        idf = defaultdict(float)
        for doc in self.tokenized_corpus:
            for token_id in set(doc):
                idf[token_id] += 1

        for token_id, doc_freq in idf.items():
            idf[token_id] = math.log(((self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1)

        return idf

    # def _calculate_term_freqs(self):
    #     term_freqs = [Counter(doc) for doc in self.tokenized_corpus]
    #     return term_freqs

    def _calculate_term_freqs(self):
        term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
        for i, doc in enumerate(self.tokenized_corpus):
            for token_id in doc:
                term_freqs[i][token_id] += 1

        return term_freqs
    
    def get_scores(self, query: str, k1:float = 1.2, b:float = 0.75):
        query  = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
        scores = np.zeros(self.n_docs)

        for q in query:
            idf = self.idf[q]
            for i, term_freq in enumerate(self.term_freqs):
                q_frequency = term_freq[q]
                doc_len     = len(self.tokenized_corpus[i])
                score_q     = idf * (q_frequency * (k1 + 1)) / ((q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
                scores[i]  += score_q

        return scores

    def get_top_k(self, query:str, k:int):
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[-k:][::-1]   # 마지막 k개의 요소를 선택하고, 역순으로 뒤집음 (=> 내림차순)
        top_k_scores  = scores[top_k_indices]
        return top_k_scores, top_k_indices