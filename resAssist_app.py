import streamlit as st
import shutil
import os

# import llama-index functions
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# setup embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# setup llm model
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Key Variables
RAG_DATA_DIR = "./data_for_RAG/test_data"
if not os.path.exists(RAG_DATA_DIR):
    os.mkdir(RAG_DATA_DIR)

# StreamLit Application ################################################3333

st.markdown(
    """
    # ResAssist (Researcher's Companion)
    RAG App for Data Science Experiments and generating insights
    """
)

uploaded_files = st.file_uploader("Create RAG Database | Upload your files", accept_multiple_files=True)
for ID, uploaded_file in enumerate(uploaded_files):
    bytes_data = uploaded_file.read()
    filename = uploaded_file.name
    save_loc = os.path.join(RAG_DATA_DIR, filename)
    with open(save_loc, 'wb') as f: 
        f.write(bytes_data)

if st.button("Create Vector Database", type="primary", use_container_width=True):

    # load documents
    st.session_state.documents = SimpleDirectoryReader(RAG_DATA_DIR).load_data()

    # create indexes for documents | uses text-embedding-ada-002 model to create vector embeddings (indexes)
    st.session_state.index = VectorStoreIndex.from_documents(st.session_state.documents, show_progress=True)

    # Building Query Engine Rather than Chat Engine | As it is a Q/A Bot
    st.session_state.query_engine = st.session_state.index.as_query_engine()

    st.write("Vector Store DB Is Ready")

searched_query = st.text_area("Ask Your Query", placeholder="Enter your text here.")
if searched_query:
    response = st.session_state.query_engine.query(searched_query)
    st.write(response.response)