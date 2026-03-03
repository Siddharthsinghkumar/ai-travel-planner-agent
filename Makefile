.PHONY: build run test

IMAGE=sidd/llm-travel-agent:latest
ENV_FILE=.env.laptopdocker

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm -p 8080:8000 \
		--env-file $(ENV_FILE) \
		--add-host=host.docker.internal:host-gateway \
		-e GIT_COMMIT=$$(git rev-parse --short HEAD) \
		$(IMAGE)

test:
	pytest -q