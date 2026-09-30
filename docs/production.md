# Production deployment

NorthStar production uses one Vercel project with two application services plus a managed PostgreSQL database:

1. the public Next.js service (`web`),
2. the internal FastAPI service (`api`),
3. a managed PostgreSQL database.

The browser talks only to Next.js. Interactive browser requests use the same-origin Next.js proxy under `/api/northstar/*`. Server-rendered pages and that proxy call FastAPI over Vercel's internal service binding, which injects `NORTHSTAR_API_URL` into the `web` service at runtime.

FastAPI is not directly exposed by the top-level Vercel rewrites.

## Vercel Services configuration

The repository root `vercel.json` defines the service graph:

- `web` -> root `web`, framework `nextjs`, public catch-all route;
- `api` -> root `.`, framework `fastapi`, entrypoint `northstar.api.app:app`;
- `web` binds to `api` through `NORTHSTAR_API_URL`.

Do not manually create `NORTHSTAR_API_URL` in Vercel. The service binding owns that variable for each local, preview, and production Services deployment.

For local multi-service development, run from the repository root:

```bash
vercel dev -L
```

Running Next.js by itself still supports the local value documented in `web/.env.example`.

## Production invariants

Production intentionally behaves more strictly than local development:

- `NORTHSTAR_ENV=production` disables the CSV data fallback;
- PostgreSQL is required for data-serving routes and readiness;
- automatic table creation defaults to off;
- FastAPI Swagger/OpenAPI pages default to off;
- the API exposes separate liveness (`/health`) and readiness (`/ready`) checks;
- API and web responses include baseline clickjacking, MIME-sniffing, referrer, permissions, and transport-security headers;
- frontend-to-API calls have a bounded timeout instead of waiting indefinitely.

## Vercel environment

Configure these project environment variables in Vercel:

```text
NORTHSTAR_ENV=production
NORTHSTAR_DATABASE_URL=<managed PostgreSQL connection string>
NORTHSTAR_ENABLE_DOCS=false
NORTHSTAR_AUTO_CREATE_SCHEMA=false
NORTHSTAR_API_TIMEOUT_MS=8000
```

`NORTHSTAR_API_URL` is intentionally absent from that list because Vercel injects it from the `web -> api` service binding.

The API is internal to the Vercel project, so direct browser CORS access is not required for the normal NorthStar web flow. If a future public FastAPI route is added, configure `NORTHSTAR_CORS_ORIGINS` explicitly for that use case.

Do not commit the production database URL or credentials.

## Load the production database

The production API does not create its schema automatically by default. Load the canonical profile explicitly before marking the application ready:

```bash
python scripts/load_profiles_to_db.py \
  --database-url "$NORTHSTAR_DATABASE_URL"
```

The loader creates the schema and replaces the selected season transactionally.

Because FastAPI stays internal, verify it through the existing Next.js proxy:

```text
GET /api/northstar/health       -> 200, status=ok
GET /api/northstar/ready        -> 200, status=ready, storage_backend=postgresql
GET /api/northstar/api/v1/meta  -> storage_backend=postgresql
```

A production deployment with no database, an unreachable database, or zero player profiles must not be considered ready.

## Deploy on Vercel

Link and deploy from the repository root, where `vercel.json` lives:

```bash
vercel link
vercel project inspect --non-interactive
vercel deploy --prod
```

Vercel builds `web` and `api` independently and deploys them atomically as one project. Preview deployments use the same service graph and binding.

## Portable API container

A standalone production image remains available if the API is ever deployed outside Vercel:

```bash
docker build -f Dockerfile.api -t northstar-api .
docker run --rm -p 8000:8000 \
  -e NORTHSTAR_ENV=production \
  -e NORTHSTAR_DATABASE_URL="$NORTHSTAR_DATABASE_URL" \
  northstar-api
```

The image runs as a non-root user, installs only the API runtime dependencies, honors `PORT` and `WEB_CONCURRENCY`, and includes a liveness health check.

## Release gate

Before every production release:

```bash
python -m pytest
ruff check src tests scripts --select E9,F63,F7,F82

cd web
npm ci
npm run lint
npm test
npm run build
npm audit --omit=dev --audit-level=high
```

CI repeats these checks and also verifies that the API Docker image builds. The frontend test suite additionally asserts the intended Vercel service graph so accidental public exposure of the FastAPI service or removal of the binding is caught in CI.

The repository still has historical Ruff formatting/import-order debt outside the production-hardening work. CI currently blocks on correctness-class Ruff rules plus the full test suite rather than mixing a large legacy formatting cleanup into deployment work.

## Post-deploy smoke test

Verify the live stack in this order:

1. `/api/northstar/health`
2. `/api/northstar/ready`
3. `/api/northstar/api/v1/meta`
4. homepage render
5. search by `Judge`, `Aaron`, `Arraez`, and MLBAM ID
6. Judge vs. Arráez comparison
7. Players, Canada, Blue Jays, Guide, and an individual player profile
8. mobile dock and mobile Show more behavior

Do not treat a successful web render alone as a successful release. The production release is healthy only when the public Next.js service, the internal FastAPI readiness path, and PostgreSQL-backed metadata all agree.
