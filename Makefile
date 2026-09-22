.PHONY: install run debug clean lint lint-strict index index-embedded index-incremental search-dataset answer-dataset evaluate moulinette

install:
	export UV_CACHE='~/goinfre'
	export HF_HOME='~/goinfre'
	uv sync
index:
	uv run python -m src index --max_chunk_size 2000
index-embedded:
	uv run python -m src index --max_chunk_size 2000 --with_embeddings
index-incremental:
	uv run python -m src index --incremental
search-dataset:
	uv run python -m src search_dataset \
    --dataset_path data/datasets_public/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
answer-dataset:
	uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
evaluate:
	uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets_public/AnsweredQuestions/dataset_docs_public.json \
    --k 10
moulinette:
	./moulinette/moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets_public/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
serve:
	uv run python -m src serve --host 127.0.0.1 --port 8000

run: install index search-dataset answer-dataset evaluate


debug:
	uv run python -m pdb -m src

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf `find . -type d -name .mypy_cache`
	rm -rf `find . -type d -name .pytest_cache`
	rm -rf data/output data/processed

lint:
	flake8 src && mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src && mypy src --strict
