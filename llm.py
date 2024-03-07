"""This file creates a Large Language Model (LLM) augmented with local information."""

import warnings

# Hide LangChainDeprecationWarning...
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import CTransformers
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate


def load_augmented_llm(llm_path: str, db_dir: str):
    llm = CTransformers(
        config={"context_length": 1000, "max_new_tokens": 512, "temperature": 0.01},
        model=str(llm_path),
        model_type="llama",
    )

    embeddings = HuggingFaceEmbeddings(
        model_kwargs={"device": "cpu"},
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    db = FAISS.load_local(db_dir, embeddings)
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

    return RetrievalQA.from_chain_type(
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
