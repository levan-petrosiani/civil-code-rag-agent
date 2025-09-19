import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEmbeddingFunction:
    def __init__(self, model="gemini-embedding-001", batch_size=100):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, input: list[str]) -> list[list[float]]:
        all_embeddings = []

        for i in range(0, len(input), self.batch_size):
            batch = input[i:i+self.batch_size]
            response = genai_client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            batch_embeddings = [e.values for e in response.embeddings]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def name(self) -> str:
        """Required by Chroma to identify embedding function."""
        return f"GeminiEmbeddingFunction-{self.model}"
