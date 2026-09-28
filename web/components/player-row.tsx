import Link from "next/link";
import { percentile, signed } from "@/lib/format";
import type { PlayerSummary } from "@/lib/types";

type Props = {
  player: PlayerSummary;
  rank?: number;
  metric?: "re24_per_100_pa" | "selection" | "contact" | "damage";
};

export function PlayerRow({ player, rank, metric = "re24_per_100_pa" }: Props) {
  const metricValue =
    metric === "selection"
      ? percentile(player.selection_context_percentile)
      : metric === "contact"
        ? percentile(player.contact_context_percentile)
        : metric === "damage"
          ? percentile(player.damage_context_percentile)
          : signed(player.re24_per_100_pa, 2);

  return (
    <Link href={`/players/${player.batter}`} className="player-row">
      <div className="player-row__rank">{rank ? String(rank).padStart(2, "0") : "↗"}</div>
      <div className="player-row__identity">
        <strong>{player.player_name}</strong>
        <span>
          {player.primary_team_2025 ?? "—"}
          {player.canadian_born ? " · Canadian-born" : ""}
        </span>
      </div>
      <div className="player-row__metric">{metricValue}</div>
      <div className="player-row__arrow">↗</div>
    </Link>
  );
}
