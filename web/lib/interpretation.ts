import type { PlayerProfile } from "@/lib/types";

function band(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "does not have enough available data for a league-relative percentile";
  }
  if (value >= 90) return "grades among the highest in the profiled 2025 hitter sample";
  if (value >= 70) return "grades above most hitters in the profiled 2025 sample";
  if (value >= 40) return "sits near the middle of the profiled 2025 hitter sample";
  if (value >= 20) return "grades below most hitters in the profiled 2025 sample";
  return "grades among the lowest in the profiled 2025 hitter sample";
}

export function playerInterpretation(player: PlayerProfile) {
  return {
    selection:
      player.player_name +
      "'s Selection profile " +
      band(player.selection_context_percentile) +
      ". This describes swing/take decision quality after accounting for pitch and game context.",
    contact:
      player.player_name +
      "'s Contact profile " +
      band(player.contact_context_percentile) +
      ". This is contact relative to the modeled difficulty of the swings taken, not raw contact rate alone.",
    damage:
      player.player_name +
      "'s Damage profile " +
      band(player.damage_context_percentile) +
      ". This describes contact quality relative to the opportunities presented.",
  };
}

export function percentilePlainEnglish(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Percentile unavailable";
  }
  const rounded = Math.max(0, Math.min(100, Math.round(value)));
  return "Higher than about " + rounded + "% of profiled hitters on this measure.";
}
