"""
This script creates a database of information gathered from local text files.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_dir: str = "test"
    """Directory containing the data ultimately used to augment the LLM."""


settings = Settings()

print(
    f"Interpreting the information of documents in the '{settings.data_dir}' directory..."
)

loader = DirectoryLoader(settings.data_dir, glob="*.txt", loader_cls=TextLoader)
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
