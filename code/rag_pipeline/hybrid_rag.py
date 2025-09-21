from rag_pipeline.sparse_retriever import SparseRetriever
from core.embeddings import GeminiEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity


def rerank_chunks(query, chunk_texts, emb_lookup,  embedding_model = GeminiEmbeddingFunction()):
    """
    ეს ფუნქცია ხელახლა აფასებს და რანჟირებს ტექსტის ფრაგმენტებს (chunk_texts) მომხმარებლის შეკითხვის (query) მიმართ, კოსინუსური მსგავსების გამოყენებით. ფუნქცია:

    1. ქმნის შეკითხვის ვექტორულ წარმოდგენას (query_emb) GeminiEmbeddingFunction-ის გამოყენებით.
    2. თითოეული ფრაგმენტისთვის იღებს ან არსებულ ემბედინგს emb_lookup-დან, ან ქმნის ახალს, თუ ის არ არსებობს.
    3. ითვლის კოსინუსურ მსგავსებას შეკითხვისა და ფრაგმენტის ემბედინგებს შორის.
    4. აბრუნებს ფრაგმენტების სიას, დალაგებულს მსგავსების ქულის მიხედვით (კლებადობით).
    """

    # Embed query 
    query_emb = embedding_model([query])[0]

    scores = []
    for text in chunk_texts:
        if text in emb_lookup:
            # Use stored embedding if available
            chunk_emb = emb_lookup[text]
        else:
            chunk_emb = embedding_model([text])[0]

        score = cosine_similarity([query_emb], [chunk_emb])[0][0]
        scores.append((score, text))

    # Sort by similarity
    ranked = [c for _, c in sorted(scores, key=lambda x: x[0], reverse=True)]
    return ranked


class HybridRAG:
    """
    ეს კლასი ახორციელებს ჰიბრიდულ RAG (Retrieval-Augmented Generation) სისტემას,
    რომელიც აერთიანებს მკვრივ (dense) და იშვიათ (sparse) ძიების მეთოდებს საქართველოს სამოქალაქო კოდექსის ტექსტის ფრაგმენტების მოსაძებნად.

    __init__(...):
    ინიციალიზაციის ფუნქცია, რომელიც იღებს Chroma კოლექციას (collection),
    ტექსტის ფრაგმენტებს (chunks) და ზღვრებს მკვრივი (top_k_dense) და იშვიათი (top_k_sparse) ძიების შედეგების რაოდენობისთვის.
    ქმნის SparseRetriever ობიექტს იშვიათი ძიებისთვის.
    
    retrieve(self, query):
    ძიების ფუნქცია, რომელიც:

    1. ახორციელებს მკვრივ ძიებას Chroma კოლექციაში, აბრუნებს top_k_dense ყველაზე შესაბამის დოკუმენტსა და მათ ემბედინგებს.
    2. ახორციელებს იშვიათ ძიებას SparseRetriever-ის გამოყენებით, აბრუნებს top_k_sparse დოკუმენტს.
    3. აერთიანებს და აშორებს დუბლიკატებს ორივე ძიების შედეგებიდან.
    4. ქმნის ემბედინგების ლუქაპს (emb_lookup) მკვრივი ძიებისთვის.
    5. ხელახლა რანჟირებს შერწყმულ დოკუმენტებს rerank_chunks ფუნქციის გამოყენებით.
    6. აბრუნებს 5 ყველაზე შესაბამის ფრაგმენტს.
    """
    
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

        # Get sparse candidates 
        sparse_docs = self.sparse.search(query, top_k=self.top_k_sparse)

        # Merge and deduplicate texts
        combined_docs = list(dict.fromkeys(dense_docs + sparse_docs))

        # Build a lookup for embeddings 
        emb_lookup = {doc: emb for doc, emb in zip(dense_docs, dense_embs)}

        # Pass both texts + embeddings to rerank
        ranked = rerank_chunks(query, combined_docs, emb_lookup)

        # return top 5 most relevant chunks
        return ranked[:5]