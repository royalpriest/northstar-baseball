import type { NextRequest } from "next/server";

const upstreamBase =
  process.env.NORTHSTAR_API_URL ?? "http://127.0.0.1:8000";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

export async function GET(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  const base = upstreamBase.endsWith("/") ? upstreamBase : `${upstreamBase}/`;
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
  } catch {
    return Response.json(
      {
        detail:
          "NorthStar API is not reachable. Make sure FastAPI is running on the configured NORTHSTAR_API_URL.",
      },
      { status: 502 },
    );
  }
}
