"""
This script creates a database of information gathered from local files.
"""

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from datetime import datetime
from humanize import naturaldelta
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_dir: str = "test"
    """Directory containing the data ultimately used to augment the LLM."""


start = datetime.now()
settings = Settings()

print(f"📂 Loading local documents...")

loader = DirectoryLoader(
    settings.data_dir, glob="*.txt", loader_cls=TextLoader, recursive=True
)

documents = loader.load()

word_loader = DirectoryLoader(
    settings.data_dir,
    glob="*.doc[x]*",
    loader_cls=UnstructuredWordDocumentLoader,
    recursive=True,
)

documents.extend(word_loader.load())

for document in documents:
    print("  📄", document.metadata["source"])

print(f"🔍 Interpreting the documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

print("💾 Saving to a local database...")
db = FAISS.from_documents(texts, embeddings)
db.save_local("bin")

print("👋 Done in", naturaldelta(datetime.now() - start))
