.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

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
