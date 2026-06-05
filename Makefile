.PHONY: install lint test run-api run-demo clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-demo:
	streamlit run src/frontend/app.py --server.port 8501

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
