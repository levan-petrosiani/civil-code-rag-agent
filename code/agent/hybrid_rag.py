from agent.sparse_retriever import SparseRetriever
import numpy as np
from embeddings.embeddings import GeminiEmbeddingFunction

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rerank_chunks(query, chunks):
    embedding_model = GeminiEmbeddingFunction()
    query_emb = embedding_model([query])[0]
    chunk_embs = embedding_model(chunks)

    scores = [cosine_similarity(query_emb, ce) for ce in chunk_embs]
    ranked_chunks = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return ranked_chunks


class HybridRAG:
    def __init__(self, collection, chunks, top_k_dense=10, top_k_sparse=10):
        self.collection = collection
        self.sparse = SparseRetriever(chunks)
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse

    def retrieve(self, query):
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k_dense
        )['documents'][0]

        sparse_results = self.sparse.search(query, top_k=self.top_k_sparse)

        # merge and deduplicate
        combined = list(dict.fromkeys(dense_results + sparse_results))

        # rerank by semantic similarity
        ranked = rerank_chunks(query, combined)

        # return top 5 most relevant chunks
        return ranked[:5]
