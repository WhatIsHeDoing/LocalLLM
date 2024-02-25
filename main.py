"""
This script reads the database of information from local text files
and uses a large language model to answer questions about their content.
"""

from datetime import datetime
from halo import Halo
from time import monotonic
import logging
import warnings

# Hide LangChainDeprecationWarning...
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import CTransformers
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

now = int(datetime.now().strftime("%Y%m%d%H%M%S"))

logging.basicConfig(
    datefmt="%Y%m%d%H%M%S",
    filename=f"logs/{now}.log",
    format="%(asctime)s:%(message)s",
    level=logging.INFO,
)

binary_directory = "bin"
"""Directory of the local binary files."""

llm_spinner = Halo("Loading the Large Language Model...", spinner="dots")
llm_spinner.start()
llm_load_start_time = monotonic()

llm = CTransformers(
    model=f"./{binary_directory}/llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama",
    config={"max_new_tokens": 256, "temperature": 0.01},
)

llm_spinner.stop_and_persist("🧠", "Large Language Model loaded")

print("📂 Loading the information interpreted from local files...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

db = FAISS.load_local(binary_directory, embeddings)

print("🔃 Preparing a version of the LLM preloaded with the local content...")
retriever = db.as_retriever(search_kwargs={"k": 2})

template = """
Use the following information to answer the question from the user.
Do not try to make up an answer if you do not know it.
Context: {context}
Question: {question}
# Only return the helpful answer below.
# Helpful answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_llm = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

print("⌛ Prepared in", round(monotonic() - llm_load_start_time), "seconds")
print("🚀 The AI chatbot is ready to answer your questions!")

while True:
    print()
    query = input("🗨️ Submit a question or task, or leave blank to exit:\n")

    if not query:
        break

    logging.info(query)
    query_spinner = Halo(text="Answering...", spinner="dots")
    query_spinner.start()
    query_start_time = monotonic()
    result = qa_llm({"query": query})["result"]

    query_spinner.stop_and_persist("💬", "The chatbot responded with:")
    print(result)
    print("⌛ Answered in", round(monotonic() - query_start_time), "seconds")
    logging.info(result)

print("👋 Closing the app...")
