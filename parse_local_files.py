"""
This script creates a database of information gathered from local files.
"""

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from datetime import datetime
from emoji import emojize
from humanize import naturaldelta
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from settings import Settings

start = datetime.now()

print(emojize(":radio_button: Loading settings..."))
settings = Settings()
print("   ", settings)

print(emojize(":open_file_folder: Loading local documents..."))

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
    print(emojize("  :page_facing_up:"), document.metadata["source"])

print(emojize(":magnifying_glass_tilted_left: Interpreting the documents..."))
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

print(emojize(":floppy_disk: Saving to a local database..."))
db = FAISS.from_documents(texts, embeddings)
db.save_local(settings.db_dir)

print(emojize(":waving_hand: Done in"), naturaldelta(datetime.now() - start))
