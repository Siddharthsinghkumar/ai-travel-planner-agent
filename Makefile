.PHONY: build run test

IMAGE=sidd/llm-travel-agent:latest

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm -p 8000:8000 \
	  --add-host=host.docker.internal:host-gateway \
	  -e OPENAI_API_KEY="" \
	  -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
	  -e WEATHER_API_KEY="$${WEATHER_API_KEY}" \
	  $(IMAGE)

test:
	pytest -q
