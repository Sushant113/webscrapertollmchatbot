import streamlit as st
import os
from scraper import scrape_website
from preprocessor import chunk_data_with_topic_modeling
from embeddings import create_embeddings, encode_bert
from milvus_handler import store_in_milvus
from retriever import query_expansion, hybrid_retrieval
from qa_model import answer_question

def main():
    st.title("CUDA Documentation Search and Q&A")

    # Check if data is already processed and stored
    if not os.path.exists("data_processed.txt"):
        with st.spinner("Processing CUDA documentation... This may take a while."):
            url = "https://docs.nvidia.com/cuda/"
            data = scrape_website(url, depth=5)
            chunks = chunk_data_with_topic_modeling(data)
            embeddings = create_embeddings(chunks)
            
            # Store embeddings using FLAT and IVF indexes
            store_in_milvus(embeddings, chunks, "FLAT")
            store_in_milvus(embeddings, chunks, "IVF_FLAT")

            # Mark data as processed
            with open("data_processed.txt", "w") as f:
                f.write("Data processed and stored in Milvus")

        st.success("CUDA documentation processed and stored successfully!")
    else:
        st.info("Using previously processed CUDA documentation.")

    user_query = st.text_input("Enter your query:")
    index_type = st.radio("Choose index type:", ("FLAT", "IVF_FLAT"))

    if st.button("Search and Answer"):
        with st.spinner("Searching and generating answer..."):
            expanded_query = query_expansion(user_query)
            results = hybrid_retrieval(expanded_query, f'cuda_docs_{index_type.lower()}', encode_bert, top_k=5)
            
            context = " ".join([result[0]["text"] for result in results])
            answer = answer_question(user_query, context)
            st.write("### Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
