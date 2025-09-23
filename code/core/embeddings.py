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
    ეს კლასი გამოიყენება ტექსტის ვექტორულ წარმოდგენად (embedding) გარდაქმნისთვის, Gemini მოდელის გამოყენებით, რაც თავსებადია Chroma მონაცემთა ბაზასთან.

    __init__(...):
        ინიციალიზაციის ფუნქცია, რომელიც ადგენს Gemini მოდელის სახელს (model) და პარტიების ზომას (batch_size)
        რომელიც განსაზღვრავს, თუ რამდენი ტექსტის ნაწილი დამუშავდება ერთდროულად.

    __call__(...):
    ძირითადი ფუნქცია, რომელიც იღებს ტექსტების სიას (input) და აბრუნებს მათ ვექტორულ წარმოდგენებს (embeddings).

        - ყოფს შეყვანილ ტექსტს პარტიებად (batch_size-ის მიხედვით).
        - თითოეული პარტიისთვის იძახებს genai_client.models.embed_content, რათა გარდაქმნას ტექსტები ვექტორებად, „SEMANTIC_SIMILARITY“ ამოცანის ტიპის გამოყენებით.
        - აგროვებს და აბრუნებს ყველა ვექტორს, როგორც სიას.


    name(self) -> str:
    აბრუნებს კლასის სახელს (GeminiEmbeddingFunction-{model}), რომელიც Chroma-ს სჭირდება ემბედინგის ფუნქციის იდენტიფიკაციისთვის.
    """

    
    class GeminiEmbeddingFunction:

        def __init__(self, model="gemini-embedding-001", batch_size=100):
            self.model = model
            self.batch_size = batch_size

        def __call__(self, input: list[str]) -> list[list[float]]:
            return self._embed(input)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embed(texts)

        def embed_query(self, input: str, **kwargs) -> list[float]:
            """Chroma passes `input` as a kwarg, so we accept it explicitly."""
            return self._embed([input])[0]

        def _embed(self, input: list[str]) -> list[list[float]]:
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
            return f"GeminiEmbeddingFunction-{self.model}"

