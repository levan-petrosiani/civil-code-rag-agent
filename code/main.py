import streamlit as st
import os
from google import genai
from dotenv import load_dotenv
from processing.text_processing import clean_noise
from processing.chunking import chunk_georgian_civil_code
from embeddings.embeddings import GeminiEmbeddingFunction
from agent.vector_store import load_data
from agent.agent import answer_question
from processing.data_processing import load_chunks
from agent.sparse_retriever import SparseRetriever
from agent.hybrid_rag import HybridRAG

load_dotenv()

@st.cache_resource
def initialize_rag_system():
    """Loads all data and initializes the RAG agent."""
    print("--- Initializing RAG System (this should run only once) ---")
    chunks = load_chunks()
    collection = load_data(chunks)

    return HybridRAG(collection, chunks)

def main():
    st.set_page_config(page_title="AI ასისტენტი", page_icon="🤖", layout="centered")
    st.title("საქართველოს სამოქალაქო კოდექსის AI RAG ასისტენტი")

    # Initialize the entire RAG system and cache it
    rag = initialize_rag_system()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("თქვენი შეკითხვა საქართველოს სამოქალაქო კოდექსის შესახებ"):
        # Store user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream response
        results = rag.retrieve(prompt)

        context = "\n\n".join(results)

        
        with st.status("ვამუშავებ პასუხს!...", expanded=True) as status:
            with st.chat_message("ai"):
                response = answer_question(prompt, context)
                # print(response)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            status.update(
                label="დასრულებულია!", state="complete", expanded=True
                )
    

if __name__ == "__main__":
    main()