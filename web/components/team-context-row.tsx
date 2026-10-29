import Link from "next/link";
import { barWidth, integer, percentileValue, signed } from "@/lib/format";
import type { PlayerSummary } from "@/lib/types";

type Props = {
  player: PlayerSummary;
  index: number;
};

export function TeamContextRow({ player, index }: Props) {
  const partial = player.primary_team_2025 !== "TOR";
  const multiTeam = player.teams_2025?.includes(";");

  return (
    <Link href={`/players/${player.batter}`} className="team-context-row">
      <div className="team-context-row__index">
        {String(index).padStart(2, "0")}
      </div>

      <div className="team-context-row__identity">
        <strong>{player.player_name}</strong>
        <span>
          {player.teams_2025 ?? player.primary_team_2025 ?? "MLB"} ·{" "}
          {integer(player.season_plate_appearances)} PA
        </span>
      </div>

      <div className="team-context-row__flags">
        {partial ? <span className="team-flag">Partial TOR</span> : null}
        {multiTeam ? <span className="team-flag team-flag--muted">Multi-team</span> : null}
        {player.canadian_born ? <span className="team-flag team-flag--canada">CAN-born</span> : null}
      </div>

      <div className="team-context-row__shape">
        <div className="team-mini-axis team-mini-axis--selection">
          <span>S</span>
          <i><b style={{ width: barWidth(player.selection_context_percentile) }} /></i>
          <strong>{percentileValue(player.selection_context_percentile)}</strong>
        </div>
        <div className="team-mini-axis team-mini-axis--contact">
          <span>C</span>
          <i><b style={{ width: barWidth(player.contact_context_percentile) }} /></i>
          <strong>{percentileValue(player.contact_context_percentile)}</strong>
        </div>
        <div className="team-mini-axis team-mini-axis--damage">
          <span>D</span>
          <i><b style={{ width: barWidth(player.damage_context_percentile) }} /></i>
          <strong>{percentileValue(player.damage_context_percentile)}</strong>
        </div>
      </div>

      <div className="team-context-row__metric">
        <span>RE24 / 100</span>
        <strong>{signed(player.re24_per_100_pa, 2)}</strong>
      </div>

      <span className="team-context-row__arrow">↗</span>
    </Link>
  );
}
