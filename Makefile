install:
	uv install

run:
	uv run ./venv/bin/python src

debug:
	pdb uv run .venv/bin/python src

lint:
	flake8 . || mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .

lint-strict:
	flake8 . || mypy --strict .

clean:
	rm -rf */__pycache__/* .mypy_chache __pycache__/
