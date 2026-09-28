import {
  CompareResponse,
  LeaderboardResponse,
  MetaResponse,
  PlayerListResponse,
  PlayerProfile,
} from "@/lib/types";

const serverBase =
  process.env.NORTHSTAR_API_URL ??
  process.env.NEXT_PUBLIC_NORTHSTAR_API_URL ??
  "http://127.0.0.1:8000";

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
  const response = await fetch(`${serverBase}${path}`, {
    cache: "no-store",
    headers: { Accept: "application/json" },
  });

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
