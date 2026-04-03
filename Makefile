PY=python

export PYTHONPATH := $(PWD)

install:
	pip install -r requirements.txt

run:
	uvicorn src.app.fastapi_app:app --host 0.0.0.0 --port $${PORT:-8000}

fmt:
	black .
	ruff check . --fix

lint:
	ruff check .
	black --check .
	bandit -r src -q

test:
	pytest -q

ingest:
	$(PY) src/ingest/ingest.py --input $(INPUT_DIR) --batch_size 200

ingest_async:
	$(PY) src/ingest/producer.py --input $(INPUT_DIR)

worker:
	$(PY) src/ingest/worker.py

eval:
	$(PY) src/eval/evaluate.py --questions ./data/eval/questions.json

docker-build:
	docker build -t rag-aerospace:local -f docker/Dockerfile .

docker-run:
	docker run --env-file .env -p 8000:8000 rag-aerospace:local

helm-dryrun:
	helm template k8s/helm
