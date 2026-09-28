import { TeamContextRow } from "@/components/team-context-row";
import { getTeamPlayers } from "@/lib/api";
import { percentile } from "@/lib/format";
import type { PlayerSummary } from "@/lib/types";

export const dynamic = "force-dynamic";

function maxBy(
  players: PlayerSummary[],
  key:
    | "selection_context_percentile"
    | "contact_context_percentile"
    | "damage_context_percentile",
) {
  return players.reduce<PlayerSummary | null>((best, player) => {
    if (!best) return player;
    return (player[key] ?? -1) > (best[key] ?? -1) ? player : best;
  }, null);
}

export default async function BlueJaysPage() {
  const players = await getTeamPlayers("TOR", 100);
  const primaryTor = players.items.filter(
    (player) => player.primary_team_2025 === "TOR",
  ).length;
  const multiTeam = players.items.filter((player) =>
    player.teams_2025?.includes(";"),
  ).length;
  const canadianBorn = players.items.filter((player) => player.canadian_born).length;

  const selectionLeader = maxBy(players.items, "selection_context_percentile");
  const contactLeader = maxBy(players.items, "contact_context_percentile");
  const damageLeader = maxBy(players.items, "damage_context_percentile");

  return (
    <div className="page-shell page-shell--narrow">
      <section className="page-hero toronto-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">Toronto Blue Jays</span>
          <h1 className="page-title">The 2025 Toronto view.</h1>
          <p>
            Every hitter with at least one Toronto plate appearance in the
            2025 dataset, with full-season NorthStar context kept distinct from
            team exposure.
          </p>
        </div>
        <div className="page-hero__aside">
          {players.count} hitters with TOR exposure
          <br />
          Full-season NorthStar diagnostics
        </div>
      </section>

      <section className="toronto-summary">
        <div className="toronto-stat">
          <span>TOR exposure</span>
          <strong>{players.count}</strong>
          <p>Hitters appearing for Toronto in 2025</p>
        </div>
        <div className="toronto-stat">
          <span>Primary TOR</span>
          <strong>{primaryTor}</strong>
          <p>Primary 2025 batting team is Toronto</p>
        </div>
        <div className="toronto-stat">
          <span>Multi-team</span>
          <strong>{multiTeam}</strong>
          <p>Profiles spanning more than one MLB team</p>
        </div>
        <div className="toronto-stat">
          <span>Canadian-born</span>
          <strong>{canadianBorn}</strong>
          <p>Birthplace-defined Canada layer</p>
        </div>
      </section>

      <section className="section toronto-context-section">
        <div className="section-header">
          <div>
            <span className="eyebrow">Context leaders</span>
            <h2>Toronto through three lenses.</h2>
          </div>
          <p>
            These are the highest league-relative NorthStar percentiles among
            hitters who appeared for Toronto, not Toronto-only split values.
          </p>
        </div>

        <div className="toronto-leader-grid">
          {selectionLeader ? (
            <a href={`/players/${selectionLeader.batter}`} className="toronto-leader-card toronto-leader-card--selection">
              <span>Selection</span>
              <strong>{percentile(selectionLeader.selection_context_percentile)}</strong>
              <p>{selectionLeader.player_name}</p>
            </a>
          ) : null}
          {contactLeader ? (
            <a href={`/players/${contactLeader.batter}`} className="toronto-leader-card toronto-leader-card--contact">
              <span>Contact</span>
              <strong>{percentile(contactLeader.contact_context_percentile)}</strong>
              <p>{contactLeader.player_name}</p>
            </a>
          ) : null}
          {damageLeader ? (
            <a href={`/players/${damageLeader.batter}`} className="toronto-leader-card toronto-leader-card--damage">
              <span>Damage</span>
              <strong>{percentile(damageLeader.damage_context_percentile)}</strong>
              <p>{damageLeader.player_name}</p>
            </a>
          ) : null}
        </div>
      </section>

      <section className="section">
        <div className="toronto-roster-panel">
          <div className="toronto-roster-panel__header">
            <div>
              <span className="eyebrow">Team context</span>
              <h2>2025 Toronto exposure.</h2>
            </div>
            <span>Ordered by full-season PA sample</span>
          </div>

          <div className="team-context-row team-context-row--head" aria-hidden="true">
            <div>No.</div>
            <div>Hitter / 2025 teams</div>
            <div>Exposure flags</div>
            <div>NorthStar shape</div>
            <div>RE24 / 100</div>
            <div />
          </div>

          {players.items.map((player, index) => (
            <TeamContextRow
              key={player.batter}
              player={player}
              index={index + 1}
            />
          ))}
        </div>
      </section>

      <div className="profile-disclaimer">
        <span className="status-dot" />
        <p>
          Team membership describes 2025 Toronto exposure. Selection, Contact,
          Damage, wOBA, xwOBA, and RE24 remain full-season MLB profile values;
          this page does not relabel them as Toronto-only splits.
        </p>
      </div>
    </div>
  );
}
