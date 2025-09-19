from rank_bm25 import BM25Okapi
import re

class SparseRetriever:
    def __init__(self, chunks):
        # Preprocess chunks into token lists
        self.documents = [chunk['text'] for chunk in chunks]
        self.tokenized_docs = [self.tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def tokenize(self, text):
        # Simple Georgian tokenizer (split by spaces; you can improve)
        text = re.sub(r'\s+', ' ', text)
        return text.split(" ")

    def search(self, query, top_k=10):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices]
    

