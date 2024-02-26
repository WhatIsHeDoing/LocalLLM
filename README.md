# Local LLM

## 👋🏻 Introduction

This repository is used to set up, augment and experiment with Large Language Models (LLMs) locally.

## 👟 Getting Started

1. First, clone an LLM such as [Llama] using `make clone_llm`.
Adjust the LLM size in the [Makefile] based on the amount of memory available on your computer.
1. Next, install Python dependencies with `make install`.
1. Augment the Llama LLM with test data by running `make parse_local_files`.
1. Run the chatbot command line app with `make run` to see it in action with the test data.```
1. Use an `.env` to customise some of the configuration of the scripts, particularly to point to your data.
Look at the instructions in [`.env.example`][env] for details.

If any of the previous steps fail, try installing the dependencies for your Operating System
using an appropriate script in the `config` directory, such as `.\config\config_windows.ps1`.

## 🔗 References

- [Scott Harden]: Using Llama 2 to Answer Questions About Local Documents

[env]: ./.env.example
[Llama]: https://llama.meta.com/
[Makefile]: ./Makefile
[Scott Harden]: https://swharden.com/blog/2023-07-30-ai-document-qa/
