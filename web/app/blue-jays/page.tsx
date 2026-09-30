import Link from "next/link";
import { TeamContextRow } from "@/components/team-context-row";
import { getTeamPlayers } from "@/lib/api";
import { percentile } from "@/lib/format";
import type { PlayerSummary } from "@/lib/types";

export const dynamic = "force-dynamic";

function maxBy(
  players: PlayerSummary[],
  key: "selection_context_percentile" | "contact_context_percentile" | "damage_context_percentile",
) {
  return players.reduce<PlayerSummary | null>((best, player) => {
    if (!best) return player;
    return (player[key] ?? -1) > (best[key] ?? -1) ? player : best;
  }, null);
}

export default async function BlueJaysPage() {
  const players = await getTeamPlayers("TOR", 100);
  const primaryTor = players.items.filter((player) => player.primary_team_2025 === "TOR").length;
  const multiTeam = players.items.filter((player) => player.teams_2025?.includes(";")).length;
  const canadianBorn = players.items.filter((player) => player.canadian_born).length;

  const leaders = [
    { key: "selection", label: "Selection", note: "Decision quality", player: maxBy(players.items, "selection_context_percentile"), value: "selection_context_percentile" as const },
    { key: "contact", label: "Contact", note: "Bat-to-ball", player: maxBy(players.items, "contact_context_percentile"), value: "contact_context_percentile" as const },
    { key: "damage", label: "Damage", note: "Impact vs opportunity", player: maxBy(players.items, "damage_context_percentile"), value: "damage_context_percentile" as const },
  ];

  return (
    <div className="page-shell page-shell--narrow index-page toronto-page--editorial">
      <section className="compact-masthead compact-masthead--split">
        <div>
          <span className="eyebrow">Toronto Blue Jays · 2025</span>
          <h1>The Toronto view.</h1>
          <p>Every hitter with Toronto exposure, while keeping full-season NorthStar context distinct from team-only performance.</p>
        </div>
        <div className="compact-summary-strip compact-summary-strip--four">
          <div><strong>{players.count}</strong><span>TOR exposure</span></div>
          <div><strong>{primaryTor}</strong><span>primary TOR</span></div>
          <div><strong>{multiTeam}</strong><span>multi-team</span></div>
          <div><strong>{canadianBorn}</strong><span>CAN-born</span></div>
        </div>
      </section>

      <section className="lens-leader-board">
        <div className="lens-leader-board__intro">
          <span className="eyebrow">Toronto through three lenses</span>
          <p>Highest league-relative NorthStar percentiles among hitters who appeared for Toronto. These remain full-season MLB values.</p>
        </div>
        <div className="lens-leader-board__rows">
          {leaders.map((leader) => leader.player ? (
            <Link key={leader.key} href={`/players/${leader.player.batter}`} className={`lens-leader-line lens-leader-line--${leader.key}`}>
              <div className="lens-leader-line__label">
                <strong>{leader.label}</strong>
                <span>{leader.note}</span>
              </div>
              <div className="lens-leader-line__value">
                <strong>{percentile(leader.player[leader.value])}</strong>
                <span>pct</span>
              </div>
              <div className="lens-leader-line__names">
                <strong>{leader.player.player_name}</strong>
                <span>{leader.player.primary_team_2025 ?? "MLB"} · open profile ↗</span>
              </div>
            </Link>
          ) : null)}
        </div>
      </section>

      <section className="editorial-index editorial-index--roster">
        <div className="editorial-index__heading">
          <div>
            <span className="eyebrow">Team context</span>
            <h2>2025 Toronto exposure</h2>
          </div>
          <span>Ordered by full-season PA sample</span>
        </div>

        <div className="team-context-row team-context-row--head editorial-row-head" aria-hidden="true">
          <div>No.</div>
          <div>Hitter / 2025 teams</div>
          <div>Exposure flags</div>
          <div>NorthStar shape</div>
          <div>RE24 / 100</div>
          <div />
        </div>

        {players.items.map((player, index) => (
          <TeamContextRow key={player.batter} player={player} index={index + 1} />
        ))}
      </section>

      <div className="editorial-note">
        <span className="status-dot" />
        <p>Team membership describes Toronto exposure. NorthStar axes, wOBA, xwOBA, and RE24 remain full-season MLB profile values.</p>
      </div>
    </div>
  );
}
