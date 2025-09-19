import chromadb
from embeddings.embeddings import GeminiEmbeddingFunction

# Persistent client (saves to disk)
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

def load_data(all_chunks):
    """
    Ensure Chroma collection is created and populated only once.
    Returns the collection object.
    """
    collection = chroma_client.get_or_create_collection(
        name="georgian_civil_code",
        embedding_function=GeminiEmbeddingFunction()
    )

    if collection.count() == 0:
        print("Collection empty. Embedding and adding chunks...")
        collection.add(
            ids=[f"chunk_{i+1}" for i in range(len(all_chunks))],
            documents=[chunk['text'] for chunk in all_chunks],
            metadatas=[chunk['metadata'] for chunk in all_chunks]
        )
        print("✅ Data added successfully.")
    else:
        print("✅ Collection already populated. Skipping embedding.")

    return collection
