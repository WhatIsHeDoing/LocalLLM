"""
This script reads the database of information curated separately from local files
and uses a Large Language Model (LLM) to answer questions about their content.
"""

from datetime import datetime
from emoji import emojize
from halo import Halo
from humanize import naturaldelta
from settings import Settings
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

print(emojize(":radio_button: Loading settings..."))
settings = Settings()
print("   ", settings)

llm_spinner = Halo("Loading the Large Language Model...", spinner="dots")
llm_spinner.start()
llm_load_start = datetime.now()

llm = CTransformers(
    model=str(settings.llm_path),
    model_type="llama",
    config={"max_new_tokens": 256, "temperature": 0.01},
)

llm_spinner.stop_and_persist(emojize(":brain:"), "Large Language Model loaded")

print(emojize(":open_file_folder: Loading local interpreted file information..."))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

db = FAISS.load_local(settings.db_dir, embeddings)

print(emojize(":plus: a version of the LLM preloaded with the local content..."))
retriever = db.as_retriever(search_kwargs={"k": 2})

template = """
Use the following information to answer the question from the user.
Do not try to make up an answer if you do not know it.
Context: {context}
Question: {question}
Only return the helpful answer below.
Helpful answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_llm = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

print(
    emojize(":hourglass_done: Prepared in"),
    naturaldelta(datetime.now() - llm_load_start),
)

print(emojize(":rocket: The AI chatbot is ready to help!"))
prompt = emojize(
    "\n:speech_balloon: Write a question or task, or leave blank to exit:\n"
)

while True:
    query = input(prompt)

    if not query:
        break

    logging.info(query)
    query_spinner = Halo(text="Answering...", spinner="dots")
    query_spinner.start()
    query_start = datetime.now()
    result = qa_llm({"query": query})  # ["result"]

    query_spinner.stop_and_persist(emojize(":robot:"), "The chatbot responded with:")
    print(result)

    print(
        emojize(":hourglass_done: Answered in"),
        naturaldelta(datetime.now() - query_start),
    )

    logging.info(result)

print(emojize(":waving_hand: Closing the app..."))
