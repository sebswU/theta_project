.PHONY: install lint test typecheck format ci

install:
	python -m pip install -e .[dev]

lint:
	ruff check .

test:
	pytest -q

typecheck:
	mypy .

format:
	ruff check . --fix

ci: lint typecheck test
