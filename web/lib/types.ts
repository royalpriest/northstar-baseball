export type PlayerSummary = {
  batter: number;
  player_name: string;
  birth_country?: string | null;
  birth_city?: string | null;
  birth_state_province?: string | null;
  primary_team_2025?: string | null;
  teams_2025?: string | null;
  season_plate_appearances?: number | null;
  plate_appearances?: number | null;
  canadian_born?: boolean | null;
  selection_context_percentile?: number | null;
  contact_context_percentile?: number | null;
  damage_context_percentile?: number | null;
  woba?: number | null;
  est_woba?: number | null;
  re24_per_100_pa?: number | null;
  [key: string]: string | number | boolean | null | undefined;
};

export type PlayerProfile = PlayerSummary & {
  eligible_decisions?: number | null;
  swings?: number | null;
  batted_balls?: number | null;
  shrunk_centered_per_100?: number | null;
  contact_above_expected_per_100_swings?: number | null;
  damage_xwoba_points_above_expected?: number | null;
  statcast_chase_rate?: number | null;
  chase_avoidance_percentile?: number | null;
  statcast_contact_rate?: number | null;
  contact_rate_percentile?: number | null;
  statcast_xwoba_on_contact?: number | null;
  xwoba_on_contact_percentile?: number | null;
  statcast_barrel_rate?: number | null;
  barrel_percentile?: number | null;
  woba_percentile?: number | null;
  xwoba_percentile?: number | null;
  re24_percentile?: number | null;
};

export type PlayerListResponse = {
  count: number;
  items: PlayerSummary[];
};

export type LeaderboardItem = PlayerSummary & {
  rank: number;
  ranked_by: string;
  scope: string;
};

export type LeaderboardResponse = {
  scope: string;
  sort_by: string;
  direction: "asc" | "desc";
  min_pa: number;
  count: number;
  items: LeaderboardItem[];
};

export type CompareResponse = {
  count: number;
  players: PlayerProfile[];
  interpretation: string;
};

export type MetaResponse = {
  season: number;
  players: number;
  storage_backend: "postgresql" | "csv";
  northstar_axes: string[];
  scope_notes: Record<string, string>;
};
