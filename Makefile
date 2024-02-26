clone_llm:
	cd ../ && \
	git clone --no-checkout --depth=1 --no-tags https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML && \
	git restore --staged llama-2-7b-chat.ggmlv3.q2_K.bin && \
	git lfs fetch --include=llama-2-7b-chat.ggmlv3.q2_K.bin

install:
	pip install -r requirements.txt

lint:
	pre-commit run --all-files

parse:
	python3 parse_local_files.py

pre_commit_setup:
	pre-commit install

pre_commit_update:
	pre-commit autoupdate

run:
	python3 main.py
