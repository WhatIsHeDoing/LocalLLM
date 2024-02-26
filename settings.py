from pathlib import Path
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Used to customise settings of the app using `.env` files."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    db_dir: DirectoryPath = Path("db")
    """Directory used to store the output of the interpretations of local data."""

    data_dir: DirectoryPath = Path("test")
    """Directory containing the local data that will be interrogated to augment the LLM."""

    llm_path: FilePath = Path("../Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q2_K.bin")
    """Relative path to the LLM used to power the chatbot."""


def print_as_dot_env(settings: Settings):
    for name, value in settings:
        print(f"  {name.upper()}={value}")
