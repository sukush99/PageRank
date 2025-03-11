run-dev:
	uv run run.py


docker-build:
	docker build -t page-rank .

docker-run:docker-build
	docker run -p 8080:8080 page-rank