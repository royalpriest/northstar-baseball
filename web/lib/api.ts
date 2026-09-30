import { connection } from "next/server";

import {
  CompareResponse,
  LeaderboardResponse,
  MetaResponse,
  PlayerListResponse,
  PlayerProfile,
} from "@/lib/types";

const DEFAULT_LOCAL_API = "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 8000;

function apiBase(): string {
  const configured = process.env.NORTHSTAR_API_URL?.trim();
  if (configured) return configured.replace(/\/$/, "");

  if (process.env.NODE_ENV === "production") {
    throw new NorthStarApiError(503, "configuration");
  }

  return DEFAULT_LOCAL_API;
}

function apiTimeoutMs(): number {
  const configured = Number(process.env.NORTHSTAR_API_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0
    ? configured
    : DEFAULT_TIMEOUT_MS;
}

export class NorthStarApiError extends Error {
  status: number;
  path: string;

  constructor(status: number, path: string) {
    super(`NorthStar API request failed (${status}) for ${path}`);
    this.name = "NorthStarApiError";
    this.status = status;
    this.path = path;
  }
}

async function request<T>(path: string): Promise<T> {
  await connection();

  let response: Response;

  try {
    response = await fetch(`${apiBase()}${path}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(apiTimeoutMs()),
    });
  } catch (error) {
    if (error instanceof NorthStarApiError) throw error;
    const status =
      error instanceof DOMException && error.name === "TimeoutError" ? 504 : 502;
    throw new NorthStarApiError(status, path);
  }

  if (!response.ok) {
    throw new NorthStarApiError(response.status, path);
  }

  return response.json() as Promise<T>;
}

export function getMeta(): Promise<MetaResponse> {
  return request<MetaResponse>("/api/v1/meta");
}

export function getPlayer(id: string | number): Promise<PlayerProfile> {
  return request<PlayerProfile>(`/api/v1/players/${id}`);
}

export function searchPlayers(query: string, limit = 8): Promise<PlayerListResponse> {
  return request<PlayerListResponse>(
    `/api/v1/players?query=${encodeURIComponent(query)}&limit=${limit}`,
  );
}

export function getTeamPlayers(team: string, limit = 100): Promise<PlayerListResponse> {
  return request<PlayerListResponse>(
    `/api/v1/teams/${encodeURIComponent(team)}?limit=${limit}`,
  );
}

export function getCanadianPlayers(limit = 100): Promise<PlayerListResponse> {
  return request<PlayerListResponse>(`/api/v1/canada?limit=${limit}`);
}

export function getLeaderboard(
  scope: string,
  sortBy: string,
  limit = 10,
  minPa = 0,
): Promise<LeaderboardResponse> {
  return request<LeaderboardResponse>(
    `/api/v1/leaderboards?scope=${encodeURIComponent(scope)}&sort_by=${encodeURIComponent(
      sortBy,
    )}&limit=${limit}&min_pa=${minPa}`,
  );
}

export function comparePlayers(players: Array<string | number>): Promise<CompareResponse> {
  const params = players
    .map((player) => `player=${encodeURIComponent(String(player))}`)
    .join("&");
  return request<CompareResponse>(`/api/v1/compare?${params}`);
}
