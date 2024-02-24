"""
This script reads the database of information from local text files
and uses a large language model to answer questions about their content.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

binary_directory = "bin"
"""Directory of the local binary files."""

# The template used when prompting the AI.
template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

print("Loading the Large Language Model...")

llm = CTransformers(
    model=f"./{binary_directory}/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    config={"max_new_tokens": 256, "temperature": 0.01},
)

print("Loading the interpreted information from the local database...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

db = FAISS.load_local("bin", embeddings)

print("Preparing a version of the LLM pre-loaded with the local content...")
retriever = db.as_retriever(search_kwargs={"k": 2})
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_llm = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

print("Asking the AI chat about information in our local files...")

query = "Who is the author of FftSharp? What is their favourite color?"
output = qa_llm({"query": query})
print(output["result"])
print(output)
