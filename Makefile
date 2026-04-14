.PHONY: install train predict lint test

PYTHON = poetry run python

install:
	@echo ">>> installing packages..."
	@poetry install --no-root

train:
	@echo ">>> training model..."
	@$(PYTHON) main.py train

predict:
	@echo ">>> generating predictions..."
	@$(PYTHON) main.py predict

lint:
	@echo ">>> linting code..."
	@poetry run ruff check .

test:
	@echo ">>> running tests..."
	@poetry run pytest