from rag_pipeline.sparse_retriever import SparseRetriever
import numpy as np
from embeddings.embeddings import GeminiEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity


def rerank_chunks(query, chunk_texts, emb_lookup,  embedding_model = GeminiEmbeddingFunction()):
   
    # Embed query ONCE
    query_emb = embedding_model([query])[0]

    scores = []
    for text in chunk_texts:
        if text in emb_lookup:
            # Use stored embedding if available
            chunk_emb = emb_lookup[text]
        else:
            # Fallback: embed if coming from BM25 only
            chunk_emb = embedding_model([text])[0]

        score = cosine_similarity([query_emb], [chunk_emb])[0][0]
        scores.append((score, text))

    # Sort by similarity
    ranked = [c for _, c in sorted(scores, key=lambda x: x[0], reverse=True)]
    return ranked


class HybridRAG:
    def __init__(self, collection, chunks, top_k_dense=10, top_k_sparse=10):
        self.collection = collection
        self.sparse = SparseRetriever(chunks)
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse

    def retrieve(self, query):
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k_dense,
            include=['documents', 'embeddings']
        )

        dense_docs = dense_results["documents"][0]
        dense_embs = dense_results["embeddings"][0]

        # Get sparse candidates (BM25 only returns text, no embeddings)
        sparse_docs = self.sparse.search(query, top_k=self.top_k_sparse)

        # Merge and deduplicate texts
        combined_docs = list(dict.fromkeys(dense_docs + sparse_docs))

        # Build a lookup for embeddings (for docs that came from dense search)
        emb_lookup = {doc: emb for doc, emb in zip(dense_docs, dense_embs)}

        # Pass both texts + embeddings to rerank
        ranked = rerank_chunks(query, combined_docs, emb_lookup)

        # return top 5 most relevant chunks
        return ranked[:5]