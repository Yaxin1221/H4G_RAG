VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: help setup ingest index refresh check ask serve clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n",$$1,$$2}'

setup:   ## create the venv and install dependencies
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "now put your key in .env (ANTHROPIC_API_KEY=...)"

ingest:  ## pull every published page from the WordPress REST API
	$(PY) -m ingest.fetch_wp

index:   ## build the country lookup + whole-page corpus
	$(PY) -m ingest.build_index

refresh: ingest index check  ## re-pull the site and rebuild (run this on a schedule)

check:   ## offline checks: lookup, aliases, search, tool output, prompt shape
	$(PY) scripts/check.py

ask:     ## ask one question from the CLI: make ask Q="who helps refugees in Jordan?"
	$(PY) scripts/ask.py "$(Q)"

serve:   ## run the web app on http://127.0.0.1:8000
	$(VENV)/bin/uvicorn app.server:app --reload --port 8000

clean:
	rm -rf data/raw data/index.json data/corpus.json
