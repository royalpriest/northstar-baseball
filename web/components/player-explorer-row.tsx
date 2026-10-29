import Link from "next/link";
import { barWidth, integer, percentile, percentileValue, signed } from "@/lib/format";
import type { PlayerSummary } from "@/lib/types";

type Lens = "re24" | "selection" | "contact" | "damage";

type Props = {
  player: PlayerSummary;
  rank: number;
  lens: Lens;
};

const lensMeta: Record<
  Lens,
  {
    label: string;
    value: (player: PlayerSummary) => string;
  }
> = {
  re24: {
    label: "RE24 / 100",
    value: (player) => signed(player.re24_per_100_pa, 2),
  },
  selection: {
    label: "Selection",
    value: (player) => percentile(player.selection_context_percentile),
  },
  contact: {
    label: "Contact",
    value: (player) => percentile(player.contact_context_percentile),
  },
  damage: {
    label: "Damage",
    value: (player) => percentile(player.damage_context_percentile),
  },
};

export function PlayerExplorerRow({ player, rank, lens }: Props) {
  const selected = lensMeta[lens];

  return (
    <Link href={`/players/${player.batter}`} className="explorer-row">
      <div className="explorer-row__rank">{String(rank).padStart(2, "0")}</div>

      <div className="explorer-row__identity">
        <strong>{player.player_name}</strong>
        <span>
          {player.primary_team_2025 ?? "—"} · {integer(player.season_plate_appearances)} PA
          {player.canadian_born ? " · Canadian-born" : ""}
        </span>
      </div>

      <div className="explorer-row__shape" aria-label="NorthStar contextual percentiles">
        <div className="explorer-mini-axis explorer-mini-axis--selection">
          <span>S</span>
          <i><b style={{ width: barWidth(player.selection_context_percentile) }} /></i>
          <strong>{percentileValue(player.selection_context_percentile)}</strong>
        </div>
        <div className="explorer-mini-axis explorer-mini-axis--contact">
          <span>C</span>
          <i><b style={{ width: barWidth(player.contact_context_percentile) }} /></i>
          <strong>{percentileValue(player.contact_context_percentile)}</strong>
        </div>
        <div className="explorer-mini-axis explorer-mini-axis--damage">
          <span>D</span>
          <i><b style={{ width: barWidth(player.damage_context_percentile) }} /></i>
          <strong>{percentileValue(player.damage_context_percentile)}</strong>
        </div>
      </div>

      <div className="explorer-row__metric">
        <span>{selected.label}</span>
        <strong>{selected.value(player)}</strong>
      </div>

      <div className="explorer-row__arrow">↗</div>
    </Link>
  );
}
