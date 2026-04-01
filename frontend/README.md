# Frontend (React + Vite)

This folder contains the runtime UI for the travel planner.

Primary app entry:
- `src/App.tsx`

Streaming + fallback client logic:
- `src/hooks/useStreamingPlan.tsx`

API base helper:
- `src/lib/api.ts`

## Local Development

```bash
cd frontend
npm install
cp .env.example .env
npm run dev -- --host 127.0.0.1 --port 5173
```

Backend expected by default:
- `http://127.0.0.1:8000`

Override with:
- `VITE_API_BASE_URL`

## Build / Preview

```bash
cd frontend
npm run build
npm run preview
```

## Environment Variables

Defined in `frontend/.env.example`:

- `VITE_API_BASE_URL`: backend API base URL
- `VITE_STREAM_SOFT_DELAY_MS`: frontend stream soft-delay marker
- `VITE_STREAM_HARD_NO_ACTIVITY_MS`: hard no-activity stream fallback timeout
- `VITE_UI_MODE`: `dev` or `preview` UX behavior
- `VITE_DEBUG_MODE`: enables debug drawer behavior in UI

## Review / Demo Artifacts

### Backend-free review artifact

```bash
cd frontend
npm run build:review-demo
```

Output:
- `../review-demo.html` (repo root)

Use case:
- share static UI/review state without backend runtime calls.

### Frozen single-file build artifact

```bash
cd frontend
npm run export:frozen-demo
```

Output:
- `dist/frozen-demo.html`

Use case:
- single-file snapshot of built frontend for static sharing/review.

## Validation Integration

Frontend runtime validation is driven from repo root by:

```bash
venv/bin/python full_validation.py --mode machine --profile full --frontend --r 0
```

The browser validator is implemented in:
- `../validation/frontend_validator.py`

Debug mode:

```bash
FRONTEND_VALIDATION_DEBUG=1 venv/bin/python ../full_validation.py --mode machine --profile full --frontend --r 0
```

## Scope Notes

- This frontend is part of the real runtime product path.
- Review/demo HTML artifacts are optional documentation/review outputs, not backend contract verification.
