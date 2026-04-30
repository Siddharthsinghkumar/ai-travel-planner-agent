# frontend/AGENTS.md

## Scope
- Frontend app rules only.
- For backend/runtime constraints, follow `../AGENTS.md`.

## Run Locally
- Install: `npm install`
- Dev: `npm run dev`
- Build: `npm run build`
- Test: `npm test`

## Frontend Validation (Playwright)
- Playwright is used only for frontend validation flows.
- Setup for validator paths:
  - `venv/bin/pip install playwright`
  - `venv/bin/playwright install chromium`
- Python validator entrypoint: `validation/frontend_validator.py`

## GPU Rule (When Triggering Backend Validation)
- If your frontend work triggers backend validation flows, check GPU first: `nvidia-smi`
- If another Python OCR/ML process is using GPU, wait

## MCP (Home-Level for OpenCode)
- Do not use `frontend/.mcp.json`.
- Use home MCP config:
  - `~/.opencode/mcp.json`
  - `~/.config/opencode/opencode.json`
- Use `claude-flow` and `claude-mem`.
- Install claude-mem if needed: `npx -y claude-mem install`
