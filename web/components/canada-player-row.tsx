import Link from "next/link";
import { birthplace, canadaAxes, hasValue } from "@/lib/canada";
import { integer, percentile, signed } from "@/lib/format";
import type { PlayerSummary } from "@/lib/types";

export function CanadaPlayerRow({ player, index }: { player: PlayerSummary; index: number }) {
  const location = birthplace(player);

  return (
    <Link href={`/players/${player.batter}`} className="canada-row">
      <span className="canada-row__index" aria-hidden="true">{String(index).padStart(2, "0")}</span>
      <div className="canada-row__identity">
        <strong>{player.player_name}</strong>
        <span>{player.teams_2025?.replaceAll(";", " / ") || player.primary_team_2025 || "MLB"} · {integer(player.season_plate_appearances)} season PA</span>
        <span className="canada-row__birthplace">Born in {location === "Birthplace detail unavailable" ? "Canada · city / province unavailable" : location}</span>
      </div>
      <div className="canada-row__shape">
        {canadaAxes.map(axis => {
          const value = player[axis.key];
          const available = hasValue(value);
          return (
            <div key={axis.key} className={`canada-axis canada-axis--${axis.tone}`} aria-label={`${axis.label}: ${available ? `${percentile(value)} league-relative percentile` : "unavailable"}`}>
              <span>{axis.label}</span>
              <i aria-hidden="true"><b style={{ width: available ? `${Math.max(0, Math.min(100, value))}%` : "0%" }} /></i>
              <strong>{available ? percentile(value) : "—"}</strong>
            </div>
          );
        })}
      </div>
      <div className="canada-row__metric">
        <span>RE24 / 100 PA</span>
        <strong>{hasValue(player.re24_per_100_pa) ? signed(player.re24_per_100_pa, 2) : "—"}</strong>
        {!hasValue(player.re24_per_100_pa) && <small>Unavailable</small>}
      </div>
      <span className="canada-row__arrow" aria-hidden="true">↗</span>
    </Link>
  );
}
