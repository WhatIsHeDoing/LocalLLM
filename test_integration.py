"""TODO Generate this test data dynamically and reference in the assertions!"""

from llm import load_augmented_llm
from parse_local_files import save_to_vector_store
from settings import Settings


def test_no_context():
    settings = Settings()

    # Deliberately using a known directory without test data...
    save_to_vector_store("test/no_context/", settings.db_dir)

    qa_llm = load_augmented_llm(settings.llm_path, settings.db_dir)

    no_context_question = "When did Winston Churchill die?"
    no_context_result: str = qa_llm({"query": no_context_question})["result"]
    assert no_context_result.find("1965") > -1

    context_question = "Who is the author of FftSharp?"
    context_result: str = qa_llm({"query": context_question})["result"]
    assert context_result.find("Scott William Harden") < 0


def test_context():
    settings = Settings()

    # Deliberately using a known directory with test data...
    save_to_vector_store("test", settings.db_dir)

    qa_llm = load_augmented_llm(settings.llm_path, settings.db_dir)

    no_context_question = "When did Winston Churchill die?"
    no_context_result: str = qa_llm({"query": no_context_question})["result"]
    assert no_context_result.find("1965") > -1

    context_question = "Who is the author of FftSharp?"
    context_result: str = qa_llm({"query": context_question})["result"]
    assert context_result.find("Scott William Harden") > -1
