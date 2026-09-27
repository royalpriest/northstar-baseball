import { getTeamPlayers } from "@/lib/api";
import { PlayerRow } from "@/components/player-row";
import { integer } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function BlueJaysPage() {
  const players = await getTeamPlayers("TOR", 100);
  const totalPa = players.items.reduce(
    (sum, player) => sum + (player.season_plate_appearances ?? 0),
    0,
  );
  const primaryTor = players.items.filter(
    (player) => player.primary_team_2025 === "TOR",
  ).length;

  return (
    <div className="page-shell page-shell--narrow">
      <section className="page-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">Toronto Blue Jays</span>
          <h1 className="page-title">The 2025 Toronto view.</h1>
          <p>
            Every hitter with at least one Toronto plate appearance in the
            2025 dataset. Traded-player team exposure is preserved instead of
            silently relabeling full-season analytics as Toronto-only.
          </p>
        </div>
        <div className="page-hero__aside">
          {players.count} hitters with TOR exposure
          <br />
          Full-season NorthStar diagnostics
        </div>
      </section>

      <section className="section">
        <div className="data-grid">
          <div className="stat-panel">
            <span className="stat-panel__label">TOR exposure</span>
            <div className="stat-panel__value">{players.count}</div>
            <div className="stat-panel__sub">Hitters appearing for Toronto</div>
          </div>
          <div className="stat-panel">
            <span className="stat-panel__label">Primary TOR</span>
            <div className="stat-panel__value">{primaryTor}</div>
            <div className="stat-panel__sub">Primary 2025 batting team is Toronto</div>
          </div>
          <div className="stat-panel">
            <span className="stat-panel__label">Profile PA</span>
            <div className="stat-panel__value">{integer(totalPa)}</div>
            <div className="stat-panel__sub">Combined full-season sample shown</div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="panel">
          <div className="panel__header">
            <h3>Toronto exposure · 2025</h3>
            <span>Historical PA membership</span>
          </div>
          {players.items.map((player, index) => (
            <PlayerRow key={player.batter} player={player} rank={index + 1} />
          ))}
        </div>
      </section>

      <div className="notice">
        NorthStar Selection, Contact, and Damage values on these profiles are
        full-season MLB diagnostics. Team membership here describes 2025
        Toronto exposure; it does not convert the metrics into Toronto-only
        splits.
      </div>
    </div>
  );
}
