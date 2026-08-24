.PHONY: help install dev test smoke check serve clip docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## install everything into the current environment
	pip install -r requirements.txt

dev: install  ## install the test tooling too
	pip install pytest httpx

test:  ## unit tests — no ffmpeg, no models, no API key
	python -m pytest tests/ -q

smoke:  ## renderer tests — needs ffmpeg with libass, ~40s
	python smoke.py

check: test smoke  ## everything CI runs

serve:  ## run the studio at http://localhost:8000
	uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

clip:  ## one-off from the CLI: make clip VIDEO=talk.mp4
	python -m pipeline.run $(VIDEO) --out out/ $(ARGS)

docker:  ## build and run the container
	docker compose up --build

clean:
	rm -rf smoke/ out/ .pytest_cache/ **/__pycache__/
