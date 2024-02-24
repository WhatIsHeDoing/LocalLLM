"""
This script creates a database of information gathered from local text files.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define what documents to load.
loader = DirectoryLoader("test/", glob="*.txt", loader_cls=TextLoader)

print("Interpreting information in the documents...")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

print("Creating the local database...")
db = FAISS.from_documents(texts, embeddings)

print("Saving the local database...")
db.save_local("bin")

print("Done!")
