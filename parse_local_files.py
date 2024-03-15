"""
This script creates a database of information gathered from local files.
"""

from datetime import datetime
from emoji import emojize
from halo import Halo
from humanize import naturaldelta

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from settings import print_as_dot_env, Settings


def save_to_vector_store(data_dir: str, db_dir: str, urls_file: str | None):
    start = datetime.now()
    urls = []

    if urls_file:
        urls = Path(urls_file).read_text().splitlines()

    loaders: list[DirectoryLoader | WebBaseLoader] = [
        WebBaseLoader(urls),
        DirectoryLoader(
            settings.data_dir, glob="*.txt", loader_cls=TextLoader, recursive=True
        ),
        DirectoryLoader(
            settings.data_dir,
            glob="*.xls[x]",
            loader_cls=UnstructuredExcelLoader,
            recursive=True,
        ),
        DirectoryLoader(
            settings.data_dir,
            glob="*.jp[e]g",
            loader_cls=UnstructuredImageLoader,
            recursive=True,
        ),
        DirectoryLoader(
            settings.data_dir,
            glob="*.png",
            loader_cls=UnstructuredImageLoader,
            recursive=True,
        ),
        DirectoryLoader(
            settings.data_dir,
            glob="*.pdf",
            loader_cls=UnstructuredPDFLoader,
            recursive=True,
        ),
        DirectoryLoader(
            settings.data_dir,
            glob="*.ppt[x]",
            loader_cls=UnstructuredPowerPointLoader,
            recursive=True,
        ),
        DirectoryLoader(
            settings.data_dir,
            glob="*.doc[x]*",
            loader_cls=UnstructuredWordDocumentLoader,
            recursive=True,
        ),
    ]

    print(emojize(":gear: Finding files..."))
    data_files = []

    for loader in loaders:
        loader_files = loader.load()
        files_detail = f"{loader.__class__.__name__}: {len(loader_files)}"

        if isinstance(loader, DirectoryLoader):
            files_detail = f"{files_detail} x {loader.glob}"

        print(emojize("  :magnifying_glass_tilted_left:"), files_detail)

        for data_file in loader_files:
            print(
                emojize("    :page_facing_up:"),
                Path(data_file.metadata["source"]).relative_to(settings.data_dir),
            )

            data_files.append(data_file)

    print(emojize(":microscope: Interpreting the documents..."))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(data_files)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    save_spinner = Halo("Saving to a local database...", spinner="dots")
    save_spinner.start()

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_dir)
    save_spinner.stop_and_persist(emojize(":floppy_disk:"), "Saved to a local database")

    print(emojize(":waving_hand: Done in"), naturaldelta(datetime.now() - start))


if __name__ == "__main__":
    print(emojize(":radio_button: Loading settings..."))
    settings = Settings()
    print_as_dot_env(settings)

    save_to_vector_store(
        str(settings.data_dir), str(settings.db_dir), settings.urls_file
    )
