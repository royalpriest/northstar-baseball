import type { PlayerSummary } from "./types";

export const canadaAxes = [
  { key: "selection_context_percentile", label: "Selection", tone: "selection", description: "Swing / take decisions in pitch context." },
  { key: "contact_context_percentile", label: "Contact", tone: "contact", description: "Bat-to-ball execution relative to expected contact." },
  { key: "damage_context_percentile", label: "Damage", tone: "damage", description: "Contact quality relative to expected damage." },
] as const;

export function hasValue(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

const provinces: Record<string, string> = {
  AB: "Alberta", BC: "British Columbia", MB: "Manitoba", NB: "New Brunswick",
  NL: "Newfoundland and Labrador", NS: "Nova Scotia", NT: "Northwest Territories",
  NU: "Nunavut", ON: "Ontario", PE: "Prince Edward Island", QC: "Quebec",
  SK: "Saskatchewan", YT: "Yukon",
};

export function birthplace(player: PlayerSummary): string {
  const province = player.birth_state_province?.trim();
  return [player.birth_city?.trim(), province ? provinces[province.toUpperCase()] ?? province : null]
    .filter(Boolean).join(", ") || "Birthplace detail unavailable";
}

export function canadaSummary(players: PlayerSummary[]) {
  const samples = players.map(p => p.season_plate_appearances).filter(hasValue);
  return {
    availableRe24: players.filter(p => hasValue(p.re24_per_100_pa)).length,
    largestSample: samples.length ? Math.max(...samples) : null,
  };
}

export function lensLeaders(players: PlayerSummary[], key: typeof canadaAxes[number]["key"]) {
  const available = players.filter(p => hasValue(p[key]));
  if (!available.length) return [];
  const maximum = Math.max(...available.map(p => p[key] as number));
  return available.filter(p => p[key] === maximum);
}
