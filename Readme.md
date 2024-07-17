# CUDA Documentation Search and Q&A System

This system allows you to search and ask questions about the CUDA documentation using a combination of web scraping, text preprocessing, embedding generation, and information retrieval techniques. The system leverages Milvus for vector storage and retrieval, as well as BERT for question answering.
## Prerequisites

- Python 3.7+
- Milvus 2.0+ (running locally or accessible)

## Setup and Installation

1. Clone the repository:
2. Create and activate a virtual environment:
3. Install the required dependencies:
4. Set up and start your Milvus server. Follow the [Milvus documentation](https://milvus.io/docs/install_standalone-docker.md) for installation instructions.

## Running the Application

1. Ensure your Milvus server is running.
2. From the project root directory, run the Streamlit application:
3. Open the provided URL in your web browser to access the application.

## Usage

1. On the first run, the application will scrape and process the CUDA documentation. This may take some time.
2. Enter your query in the text input field.
3. Choose the index type (FLAT or IVF_FLAT).
4. Click "Search and Answer" to retrieve relevant documents and generate an answer.

## Components

- `scraper.py`: Web scraping module to fetch CUDA documentation.
- `preprocessor.py`: Text preprocessing and chunking using topic modeling.
- `embeddings.py`: Creating embeddings for text chunks using BERT and SentenceTransformers.
- `milvus_handler.py`: Handling storage and indexing of embeddings in Milvus.
- `retriever.py`: Implementing hybrid retrieval combining BM25 and BERT-based similarity.
- `qa_model.py`: Question-answering model using a pre-trained pipeline.
- `main.py`: Main script to run the Streamlit application.

## Customization

You can modify parameters such as the number of topics, chunk size, or retrieval weights in the respective Python files to fine-tune the system's performance.


## Refrences
* [Milvus](https://milvus.io/)
* [Huggingface](https://huggingface.co/)
* [CUDA](https://docs.nvidia.com/cuda/)
