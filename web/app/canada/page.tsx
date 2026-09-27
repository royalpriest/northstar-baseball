import { getCanadianPlayers, getLeaderboard } from "@/lib/api";
import { PlayerRow } from "@/components/player-row";
import { integer } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function CanadaPage() {
  const [players, leaderboard] = await Promise.all([
    getCanadianPlayers(100),
    getLeaderboard("canadian", "re24_per_100_pa", 25),
  ]);

  return (
    <div className="page-shell page-shell--narrow">
      <section className="page-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">Canada layer</span>
          <h1 className="page-title">Canadian-born hitters.</h1>
          <p>
            A birthplace-defined view of the 2025 MLB profile. NorthStar does
            not infer citizenship or national-team eligibility from this
            dataset.
          </p>
        </div>
        <div className="page-hero__aside">
          {players.count} Canadian-born profiles
          <br />
          Birthplace, not citizenship
        </div>
      </section>

      <section className="section">
        <div className="data-grid">
          <div className="stat-panel">
            <span className="stat-panel__label">Profiles</span>
            <div className="stat-panel__value">{players.count}</div>
            <div className="stat-panel__sub">Canadian-born hitters in NorthStar</div>
          </div>
          <div className="stat-panel">
            <span className="stat-panel__label">Qualified RE24</span>
            <div className="stat-panel__value">{leaderboard.count}</div>
            <div className="stat-panel__sub">Profiles with RE24 / 100 PA available</div>
          </div>
          <div className="stat-panel">
            <span className="stat-panel__label">Largest sample</span>
            <div className="stat-panel__value">
              {integer(
                Math.max(
                  ...players.items.map(
                    (player) => player.season_plate_appearances ?? 0,
                  ),
                ),
              )}
            </div>
            <div className="stat-panel__sub">Season plate appearances</div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="panel">
          <div className="panel__header">
            <h3>All Canadian-born profiles</h3>
            <span>Sorted by season sample</span>
          </div>
          {players.items.map((player, index) => (
            <PlayerRow key={player.batter} player={player} rank={index + 1} />
          ))}
        </div>
      </section>

      <div className="notice">
        “Canadian-born” means MLB birth country equals Canada. It is an
        automated birthplace filter and should not be interpreted as a
        citizenship or nationality claim.
      </div>
    </div>
  );
}
