import type { NextRequest } from "next/server";

const DEFAULT_LOCAL_API = "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 8000;

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

function upstreamBase(): string | null {
  const configured = process.env.NORTHSTAR_API_URL?.trim();
  if (configured) return configured;

  return process.env.NODE_ENV === "production" ? null : DEFAULT_LOCAL_API;
}

function upstreamTimeoutMs(): number {
  const configured = Number(process.env.NORTHSTAR_API_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0
    ? configured
    : DEFAULT_TIMEOUT_MS;
}

export async function GET(request: NextRequest, context: RouteContext) {
  const configuredBase = upstreamBase();
  if (!configuredBase) {
    return Response.json(
      { detail: "NorthStar API is not configured." },
      { status: 503 },
    );
  }

  const { path } = await context.params;
  const base = configuredBase.endsWith("/") ? configuredBase : `${configuredBase}/`;
  const upstream = new URL(
    path.map((segment) => encodeURIComponent(segment)).join("/"),
    base,
  );
  upstream.search = request.nextUrl.search;

  try {
    const response = await fetch(upstream, {
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
      signal: AbortSignal.timeout(upstreamTimeoutMs()),
    });

    const body = await response.arrayBuffer();
    return new Response(body, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("content-type") ??
          "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    const timedOut =
      error instanceof DOMException && error.name === "TimeoutError";

    return Response.json(
      {
        detail: timedOut
          ? "NorthStar API timed out."
          : "NorthStar API is temporarily unavailable.",
      },
      { status: timedOut ? 504 : 502 },
    );
  }
}
