try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
    print("✅ Using pysqlite3-binary for modern SQLite support")
except ImportError:
    print("⚠️ pysqlite3-binary not installed, relying on system SQLite")

import chromadb
from core.embeddings import GeminiEmbeddingFunction

# Persistent client (ინახება მეხსიერებაში)
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

def load_data(all_chunks):
    """
    ეს ფუნქცია ქმნის ან იღებს Chroma მონაცემთა ბაზის კოლექციას სახელად „georgian_civil_code“ და ავსებს მას მოწოდებული ტექსტის ფრაგმენტებით (all_chunks),
    თუ ის ცარიელია. ფუნქცია იყენებს GeminiEmbeddingFunction-ს ტექსტის ემბედინგისთვის (ვექტორულ წარმოდგენად გარდაქმნისთვის).
    თუ კოლექცია ცარიელია, ის ამატებს ფრაგმენტების ID-ებს, ტექსტებსა და მეტამონაცემებს.
    თუ კოლექცია უკვე შევსებულია, გამოტოვებს ამ ნაბიჯს. აბრუნებს კოლექციის ობიექტს.
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