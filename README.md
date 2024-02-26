# Local LLM

## 👋🏻 Introduction

This repository is used to set up, augment and experiment with Large Language Models (LLMs) locally.

## 👟 Getting Started

1. First, clone an LLM such as [Llama] using `make clone_llm`.
Adjust the LLM size in the [Makefile] based on the amount of memory available on your computer.
1. Next, install Python dependencies with `make install`.
1. Parse the test data used to augment the LLM by running `make parse`.
1. Run the chatbot command line app with `make run` to see it in action with the test data.```

If any of the previous steps fail, try installing the dependencies for your Operating System
using an appropriate script in the `config` directory, such as `.\config\config_windows.ps1`.

## 🔧 Customising

Use an `.env` file to customise some of the configuration of the scripts, particularly to point to your data.
Look at the instructions in [`.env.example`][env] for details.

## 🆘 Contributing

All contributions are greatly appreciated! Please add the [pre-commit] hooks
using `make pre_commit_setup` to run quality checks on the codebase before submitting a Pull Request.
These can be run ad hoc with `make lint`.

## 🔗 References

- [Scott Harden]: Using Llama 2 to Answer Questions About Local Documents

[env]: ./.env.example
[Llama]: https://llama.meta.com/
[Makefile]: ./Makefile
[pre-commit]: https://pre-commit.com/
[Scott Harden]: https://swharden.com/blog/2023-07-30-ai-document-qa/
