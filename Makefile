.PHONY: install test train serve build run clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

train:
	python -m src.models.train

serve:
	python -m uvicorn src.app.main:app --reload

build:
	docker build -t bike-rental .

run:
	docker run -p 8000:8000	bike-rental

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +