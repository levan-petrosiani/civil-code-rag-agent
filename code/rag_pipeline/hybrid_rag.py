from rag_pipeline.sparse_retriever import SparseRetriever
from core.embeddings import GeminiEmbeddingFunction
import numpy as np

def _normalize_emb(x):
    """Convert numpy arrays -> python lists; leave lists as-is."""
    if hasattr(x, "tolist"):
        return x.tolist()
    return x

def _cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def rerank_chunks(query, chunk_texts, emb_lookup, embedding_model=GeminiEmbeddingFunction(), query_emb=None):
    # Reuse query_emb if provided
    if query_emb is None:
        query_emb = embedding_model([query])[0]
    query_emb = _normalize_emb(query_emb)

    scores = []
    for text in chunk_texts:
        chunk_emb = emb_lookup.get(text)
        if chunk_emb is None:
            chunk_emb = embedding_model([text])[0]
        chunk_emb = _normalize_emb(chunk_emb)
        score = _cosine_similarity(query_emb, chunk_emb)
        scores.append((score, text))

    ranked = [c for _, c in sorted(scores, key=lambda x: x[0], reverse=True)]
    return ranked

class HybridRAG:
    def __init__(self, collection, chunks, top_k_dense=10, top_k_sparse=10):
        self.collection = collection
        self.sparse = SparseRetriever(chunks)
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.embedding_model = GeminiEmbeddingFunction()

    def retrieve(self, query):
        # compute query embedding once and pass it to Chroma
        query_emb = self.embedding_model([query])[0]

        dense_results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=self.top_k_dense,
            include=['documents', 'embeddings']
        )

        dense_docs = dense_results.get("documents", [[]])[0]
        dense_embs = dense_results.get("embeddings", [[]])[0]

        # Normalize numpy -> list (Chroma may return numpy arrays on cloud)
        if hasattr(dense_embs, "tolist"):
            dense_embs = dense_embs.tolist()

        # Fallback: if embeddings are missing or not in expected structure
        if len(dense_embs) == 0 or isinstance(dense_embs[0], float):
            dense_embs = self.embedding_model(dense_docs)

        sparse_docs = self.sparse.search(query, top_k=self.top_k_sparse)
        combined_docs = list(dict.fromkeys(dense_docs + sparse_docs))
        emb_lookup = {doc: emb for doc, emb in zip(dense_docs, dense_embs)}

        ranked = rerank_chunks(query, combined_docs, emb_lookup, self.embedding_model, query_emb)
        return ranked[:5]
