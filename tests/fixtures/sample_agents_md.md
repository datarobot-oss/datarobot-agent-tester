<!-- AGENTS:GENERATED:START -->
# my-service

A lightweight HTTP API built with FastAPI that serves model predictions.

## Key files
- `main.py` — FastAPI app and route definitions
- `model.py` — Model loading and inference logic
- `tests/` — pytest suite

## Commands
- Install: `uv sync --dev`
- Test: `uv run pytest`
- Dev server: `uv run uvicorn main:app --reload`
- Lint: `uv run ruff check . --fix`

## Conventions
- All routes return `{"result": ..., "error": null}` or `{"result": null, "error": "..."}`
- Model is loaded once at startup via `@app.on_event("startup")`
- Do NOT modify `model_weights/` — these are generated artifacts
<!-- AGENTS:GENERATED:END -->

<!-- Add custom content below this line. It will be preserved when AGENTS.md is regenerated. -->
