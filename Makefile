ALPHAS = 0.0001 0.0005 0.001 0.002 0.005 0.01 0.02 0.025 0.03 0.04 0.05 0.06 0.075 0.1 0.125 0.15 0.2 0.25

.PHONY: lint test docs build ci ci-docker tables validate benchmark
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pydoclint src

test:
	uv run pytest --cov --cov-report=term-missing

docs:
	uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

build:
	uv build
	uv run twine check dist/*

ci: lint test docs build

ci-docker:
	docker run --rm -v "$(CURDIR):/work" -w /work -e UV_PROJECT_ENVIRONMENT=/tmp/tf-venv python:3.12 sh -c "pip install uv && uv sync --frozen --all-groups && make ci"

tables:
	uv run tf-inference generate --alpha $(ALPHAS) --output src/tf_inference/data

validate:
	uv run python validation/validate.py --output validation/results.json

benchmark:
	uv run python benchmarks/benchmark.py --output benchmarks/results.json
