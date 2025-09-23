import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = st.secrets["GOOGLE_API_KEY"]
genai_client = genai.Client(api_key=api_key)

class GeminiEmbeddingFunction:
    """
    Fully Chroma-compatible embedding function for Gemini embeddings.
    """

    def __init__(self, model="gemini-embedding-001", batch_size=100):
        self.model = model
        self.batch_size = batch_size

    # THIS MUST EXACTLY MATCH CHROMA INTERFACE
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    # Called by Chroma for embedding multiple documents
    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    # Called by Chroma for embedding a single query
    def embed_query(self, input: str, **kwargs) -> list[float]:
        embs = self._embed([input])
        return embs[0]

    # Internal function that calls Gemini API
    def _embed(self, input: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]
            response = genai_client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            all_embeddings.extend([list(e.values) for e in response.embeddings])
        return all_embeddings

    # Required by Chroma for collection creation
    def name(self) -> str:
        return f"GeminiEmbeddingFunction-{self.model}"


