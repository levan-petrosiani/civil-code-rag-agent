import streamlit as st
from dotenv import load_dotenv
from rag_pipeline.vector_store import load_data
from rag_pipeline.llm import answer_question
from processing.data_processing import load_chunks
from rag_pipeline.hybrid_rag import HybridRAG

load_dotenv()

@st.cache_resource
def initialize_rag_system():
    """RAG სისტემის ინიციალიზაცია"""
    chunks = load_chunks()
    collection = load_data(chunks)

    return HybridRAG(collection, chunks)

def main():
    """
    ეს ფუნქცია წარმოადგენს Streamlit აპლიკაციის ძირითად ლოგიკას, რომელიც ქმნის ინტერაქტიულ AI ასისტენტს საქართველოს სამოქალაქო კოდექსისთვის.
    """

    st.set_page_config(page_title="AI ასისტენტი", page_icon="🤖", layout="centered")
    st.title("საქართველოს სამოქალაქო კოდექსის AI RAG ასისტენტი")

    welcome = """
    👋 გამარჯობა!\n
    მე ვარ **AI იურიდიული ასისტენტი**, რომელიც სპეციალიზებულია **საქართველოს სამოქალაქო კოდექსში**.\n
    შეგიძლიათ მომწეროთ ნებისმიერი კითხვა და მე მოგაწვდით ზუსტ და სტრუქტურირებულ პასუხს **მხოლოდ კოდექსის ტექსტზე დაყრდნობით.**    
    """

    # Initialize the entire RAG system and cache it
    rag = initialize_rag_system()

    with st.chat_message("ai"):
        st.write(welcome)



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