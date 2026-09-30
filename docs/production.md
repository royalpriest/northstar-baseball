# Production deployment

NorthStar production is a three-service application:

1. a Next.js web application,
2. a FastAPI service,
3. a managed PostgreSQL database.

The browser talks to the same-origin Next.js proxy under `/api/northstar/*`. The proxy and server-rendered pages then call FastAPI using the private/server-side `NORTHSTAR_API_URL`.

## Production invariants

Production intentionally behaves more strictly than local development:

- `NORTHSTAR_ENV=production` disables the CSV data fallback.
- PostgreSQL is required for data-serving routes and readiness.
- automatic table creation defaults to off;
- FastAPI Swagger/OpenAPI pages default to off;
- the API exposes separate liveness (`/health`) and readiness (`/ready`) checks;
- API and web responses include baseline clickjacking, MIME-sniffing, referrer, and permissions headers;
- frontend-to-API calls have a bounded timeout instead of waiting indefinitely.

## Backend environment

Set these values on the deployed FastAPI service:

```text
NORTHSTAR_ENV=production
NORTHSTAR_DATABASE_URL=<managed PostgreSQL connection string>
NORTHSTAR_CORS_ORIGINS=https://your-web-domain.example
NORTHSTAR_ENABLE_DOCS=false
NORTHSTAR_AUTO_CREATE_SCHEMA=false
WEB_CONCURRENCY=2
```

The platform normally provides `PORT`. Do not commit the production database URL or credentials.

If direct browser access to FastAPI is not needed, CORS can remain restricted. NorthStar's interactive browser requests normally go through the Next.js same-origin proxy.

## Frontend environment

Set these values on the deployed Next.js service:

```text
NORTHSTAR_API_URL=https://your-fastapi-service.example
NORTHSTAR_API_TIMEOUT_MS=8000
```

`NORTHSTAR_API_URL` is intentionally server-side. Do not replace it with a public `NEXT_PUBLIC_*` variable.

## Load the production database

The production API does not create its schema automatically by default. Load the canonical profile explicitly before marking the API ready:

```bash
python scripts/load_profiles_to_db.py \
  --database-url "$NORTHSTAR_DATABASE_URL"
```

The loader creates the schema and replaces the selected season transactionally.

After loading, verify:

```text
GET /health  -> 200, status=ok
GET /ready   -> 200, status=ready, storage_backend=postgresql
GET /api/v1/meta -> storage_backend=postgresql
```

A production deployment with no database, an unreachable database, or zero player profiles must not be considered ready.

## API container

A portable production image is available:

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

CI repeats these checks and also verifies that the API Docker image builds.

The repository still has historical Ruff formatting/import-order debt outside the production-hardening work. CI currently blocks on correctness-class Ruff rules plus the full test suite rather than pretending the entire legacy tree is style-clean. A repository-wide Ruff cleanup can be done as a separate, reviewable change.

## Post-deploy smoke test

Verify the live stack in this order:

1. FastAPI `/health`
2. FastAPI `/ready`
3. FastAPI `/api/v1/meta`
4. homepage render
5. search by `Judge`, `Aaron`, `Arraez`, and MLBAM ID
6. Judge vs. Arráez comparison
7. Players, Canada, Blue Jays, Guide, and an individual player profile
8. mobile dock and mobile Show more behavior

Do not treat a successful web deployment alone as a successful release. The production release is healthy only when the web app, API readiness check, and PostgreSQL-backed metadata all agree.
