import Link from "next/link";
import { getCanadianPlayers } from "@/lib/api";
import { CanadaPlayerRow } from "@/components/canada-player-row";
import { canadaAxes, canadaSummary, lensLeaders } from "@/lib/canada";
import { integer, percentile } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function CanadaPage() {
  const players = await getCanadianPlayers(100);
  const summary = canadaSummary(players.items);

  return (
    <div className="page-shell page-shell--narrow canada-page">
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

      <section className="canada-summary" aria-label="Canadian-born profile summary">
        <div className="stat-panel">
          <span className="stat-panel__label">Canadian-born profiles</span>
          <div className="stat-panel__value">{players.items.length}</div>
          <p className="stat-panel__sub">Hitters in this 2025 view, defined by MLB birth country.</p>
        </div>
        <div className="stat-panel">
          <span className="stat-panel__label">RE24 coverage</span>
          <div className="stat-panel__value">{summary.availableRe24}<small> / {players.items.length}</small></div>
          <p className="stat-panel__sub">Profiles with RE24 / 100 PA available. <a href="#canada-re24">Why values may be missing ↗</a></p>
        </div>
        <div className="stat-panel">
          <span className="stat-panel__label">Largest season sample</span>
          <div className="stat-panel__value">{integer(summary.largestSample)}<small> PA</small></div>
          <p className="stat-panel__sub">Most full-season plate appearances among these hitters.</p>
        </div>
      </section>

      <section className="section">
        <div className="section-header">
          <div><span className="eyebrow">Context leaders</span><h2>Canada through three lenses.</h2></div>
          <p>Highest available league-relative percentiles in this Canadian-born group. Three separate diagnostics, never a composite score or a Canada-only percentile.</p>
        </div>
        <div className="canada-lenses">
          {canadaAxes.map(axis => {
            const leaders = lensLeaders(players.items, axis.key);
            return (
              <article key={axis.key} className={`toronto-leader-card toronto-leader-card--${axis.tone} canada-lens`}>
                <span>{axis.label}</span>
                <strong>{leaders.length ? percentile(leaders[0][axis.key]) : "—"}</strong>
                <span className="canada-lens__unit">League-relative percentile</span>
                <div className="canada-lens__names">{leaders.length ? leaders.map(player => <Link key={player.batter} href={`/players/${player.batter}`}>{player.player_name} ↗</Link>) : "No available values"}</div>
                <p>{axis.description}</p>
              </article>
            );
          })}
        </div>
      </section>

      <section className="section">
        <div className="toronto-roster-panel">
          <div className="toronto-roster-panel__header canada-roster-heading">
            <div><span className="eyebrow">Birthplace & batting context</span><h2>All Canadian-born profiles</h2></div>
            <span>Ordered by season PA · not an overall ranking</span>
          </div>
          <p className="canada-roster-key">Selection / Contact / Damage bars show league-relative percentiles. A dash means unavailable; it does not mean zero. Open any hitter for the full profile.</p>
          {players.items.length ? players.items.map((player, index) => <CanadaPlayerRow key={player.batter} player={player} index={index + 1} />) : <p className="canada-empty">No Canadian-born profiles are available in this dataset.</p>}
        </div>
      </section>

      <section className="canada-coverage" id="canada-re24" aria-labelledby="canada-re24-title">
        <span className="eyebrow">Reading the gaps</span>
        <h2 id="canada-re24-title">Why is RE24 sometimes missing?</h2>
        <p>RE24 describes the change in run expectancy across a plate appearance, including runs scored. NorthStar joins it from a separate run-value validation table. A hitter can have Selection, Contact, and Damage context without an available RE24 value in that source.</p>
        <p>A dash means the value is unavailable in the current profile—not zero runs, a poor result, or exclusion from the Canadian-born group. Season PA and RE24 source coverage are different things; this view does not infer a specific reason for an individual missing value.</p>
      </section>

      <div className="profile-disclaimer">
        <span className="status-dot" />
        <p>“Canadian-born” means MLB birth country equals Canada. Birthplace does not establish citizenship, nationality, or national-team eligibility. All batting values remain full-season MLB context, and Selection, Contact, and Damage remain separate.</p>
      </div>
    </div>
  );
}
