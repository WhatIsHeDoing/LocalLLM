# Local LLM

## 👋🏻 Introduction

This repository is used to set up, augment and experiment with Large Language Models (LLMs) locally.

## 👟 Getting Started

1. First, [download] a [Llama] LLM and add it to the `bin` directory.
Choose the size appropriate for the amount of memory available on your computer.
1. Next, with Python installed, run the following [Makefile] commands to install dependencies,
augment the Llama LLM with test data and run a simple query using it.

    ```sh
    make install
    make parse_local_files
    make run
    ```

1. If any of the previous steps fail, you can try installing the dependencies for your Operating System
using an appropriate script in the `config` directory, such as `.\config\config_windows.ps1`.
1. A `.env` file can be used to customise some of the configuration of the scripts.
Look at the instructions in [`.env.example`][env] for details.

## 🔗 References

- [Scott Harden]: Using Llama 2 to Answer Questions About Local Documents

[download]: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
[env]: ./.env.example
[Llama]: https://llama.meta.com/
[Makefile]: ./Makefile
[Scott Harden]: https://swharden.com/blog/2023-07-30-ai-document-qa/
