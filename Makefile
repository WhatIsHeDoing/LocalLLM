llama_git_repo_name := Llama-2-7B-Chat-GGML
vigogne_git_repo_name := Vigogne-2-13B-Instruct-GGUF

clean:
	rm db/index.* logs/*.log

clone_llama:
	cd ../ && \
	git clone --no-checkout --depth=1 --no-tags https://huggingface.co/TheBloke/$(llama_git_repo_name) && \
	cd $(llama_git_repo_name) && \
	git lfs pull --include llama-2-7b-chat.ggmlv3.q2_K.bin

clone_vigogne:
	cd ../ && \
	git clone --no-checkout --depth=1 --no-tags https://huggingface.co/TheBloke/$(vigogne_git_repo_name) && \
	cd $(vigogne_git_repo_name) && \
	git lfs pull --include vigogne-2-13b-instruct.Q4_K_M.gguf

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

tests:
	pytest
