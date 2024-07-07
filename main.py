import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("Research Article Explorer Generative Artificial Intelligence")
st.sidebar.title("📄 Article URLs")

# Function to add a new URL input field
def add_url_input():
    urls.append("")
    st.session_state.url_count += 1

# Initialize URL count
if 'url_count' not in st.session_state:
    st.session_state.url_count = 1

urls = []
for i in range(st.session_state.url_count):
    url = st.sidebar.text_input(f"🔗 URL {i + 1}", key=f"url_{i}")
    urls.append(url)

st.sidebar.button("Add URL ➕", on_click=add_url_input)
process_url_clicked = st.sidebar.button("Process URLs 🚀")

file_path = "faiss_store_hf.pkl"
main_placeholder = st.empty()

llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key="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"
)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    with st.spinner("📥 Data Loading..."):
        data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ',', ' '],
        chunk_size=1000
    )
    with st.spinner("🔍 Splitting Text..."):
        docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Adjust model_name as needed
    with st.spinner("🔧 Building Embedding Vectors..."):
        vectorstore_hf = FAISS.from_documents(docs, embeddings)
        time.sleep(2)  # Simulate some processing time

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("📝 Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
