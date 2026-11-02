PYTHON ?= python
ALPHAS = 0.0001 0.0005 0.001 0.002 0.005 0.01 0.02 0.025 0.03 0.04 0.05 0.06 0.075 0.1 0.125 0.15 0.2 0.25

.PHONY: lint test docs build ci ci-docker tables validate benchmark
lint:
	$(PYTHON) -m black --check src tests validation benchmarks
	$(PYTHON) -m isort --check-only src tests validation benchmarks
	$(PYTHON) -m flake8 src tests validation benchmarks
	$(PYTHON) -m ruff check src tests validation benchmarks

test:
	$(PYTHON) -m pytest

docs:
	$(PYTHON) -m sphinx -W --keep-going -b html docs docs/_build/html

build:
	$(PYTHON) -m build --no-isolation

ci: lint test docs build

ci-docker:
	docker run --rm -v "$(CURDIR):/work" -w /work python:3.11 sh -c "pip install -e '.[dev]' && make ci"

tables:
	$(PYTHON) -m tf_inference.cli generate --alpha $(ALPHAS) --output src/tf_inference/data

validate:
	$(PYTHON) validation/validate.py --output validation/results.json

benchmark:
	$(PYTHON) benchmarks/benchmark.py --output benchmarks/results.json
