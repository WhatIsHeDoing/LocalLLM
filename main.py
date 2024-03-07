"""
This script uses an augmented Large Language Model (LLM) to answer questions or tasks.
"""

from datetime import datetime
from emoji import emojize
from halo import Halo
from humanize import naturaldelta
from llm import load_augmented_llm
from settings import print_as_dot_env, Settings
import logging


if __name__ == "__main__":
    now = int(datetime.now().strftime("%Y%m%d%H%M%S"))

    logging.basicConfig(
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        filename=f"logs/{now}.log",
        format="%(asctime)s | %(message)s",
        level=logging.INFO,
    )

    print(emojize(":radio_button: Loading settings..."))
    settings = Settings()
    print_as_dot_env(settings)

    llm_spinner = Halo(
        "Loading and augmenting the Large Language Model...", spinner="dots"
    )

    llm_spinner.start()
    llm_load_start = datetime.now()

    qa_llm = load_augmented_llm(settings.llm_path, settings.db_dir)

    llm_spinner.stop_and_persist(
        emojize(":brain:"), "Augmented Large Language Model loaded"
    )

    print(
        emojize(":hourglass_done: Ready in"),
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
        result = qa_llm({"query": query})["result"]

        query_spinner.stop_and_persist(emojize(":robot:"), "The chatbot says:")
        print(result)

        print(
            emojize(":hourglass_done: Answered in"),
            naturaldelta(datetime.now() - query_start),
        )

        logging.info(result)

    print(emojize(":waving_hand: Closing the app..."))
