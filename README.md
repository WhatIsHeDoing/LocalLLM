# Local LLM

## Introduction

This repository is used to set up, augment and experiment with Large Language Models (LLMs) locally.

## Getting Started

1. First, [download] the Llama LLM and add it to the `bin` directory.
1. Next, with Python installed, run the following [Makefile] commands to install dependencies,
augment the [Llama] LLM with test data and run a simple query using it.

    ```sh
    make install
    make parse_local_files
    make run
    ```

## References

- [Scott Harden]: Using Llama 2 to Answer Questions About Local Documents

[download]: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
[Llama]: https://llama.meta.com/
[Makefile]: ./Makefile
[Scott Harden]: https://swharden.com/blog/2023-07-30-ai-document-qa/
