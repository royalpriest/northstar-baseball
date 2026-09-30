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
    <div className="page-shell page-shell--narrow index-page canada-page canada-page--editorial">
      <section className="compact-masthead compact-masthead--split">
        <div>
          <span className="eyebrow">Canada layer · 2025</span>
          <h1>Canadian-born hitters.</h1>
          <p>Birthplace-defined MLB context. NorthStar does not infer citizenship or national-team eligibility.</p>
        </div>
        <div className="compact-summary-strip">
          <div><strong>{players.items.length}</strong><span>profiles</span></div>
          <div><strong>{summary.availableRe24}</strong><span>RE24 values</span></div>
          <div><strong>{integer(summary.largestSample)}</strong><span>largest PA sample</span></div>
        </div>
      </section>

      <section className="lens-leader-board">
        <div className="lens-leader-board__intro">
          <span className="eyebrow">Canada through three lenses</span>
          <p>Highest available league-relative percentiles among Canadian-born hitters. These are separate diagnostics, not a composite ranking.</p>
        </div>

        <div className="lens-leader-board__rows">
          {canadaAxes.map((axis) => {
            const leaders = lensLeaders(players.items, axis.key);
            const leader = leaders[0];
            return (
              <div key={axis.key} className={`lens-leader-line lens-leader-line--${axis.tone}`}>
                <div className="lens-leader-line__label">
                  <strong>{axis.label}</strong>
                  <span>{axis.description}</span>
                </div>
                <div className="lens-leader-line__value">
                  <strong>{leader ? percentile(leader[axis.key]) : "—"}</strong>
                  <span>pct</span>
                </div>
                <div className="lens-leader-line__names">
                  {leaders.length ? leaders.map((player) => (
                    <Link key={player.batter} href={`/players/${player.batter}`}>{player.player_name} ↗</Link>
                  )) : <span>No available values</span>}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="editorial-index editorial-index--roster">
        <div className="editorial-index__heading">
          <div>
            <span className="eyebrow">Birthplace & batting context</span>
            <h2>All Canadian-born profiles</h2>
          </div>
          <span>Season PA order · not an overall ranking</span>
        </div>
        <p className="editorial-index__key">Selection / Contact / Damage use league-relative percentiles. A dash means unavailable, not zero.</p>
        {players.items.length ? players.items.map((player, index) => (
          <CanadaPlayerRow key={player.batter} player={player} index={index + 1} />
        )) : <p className="canada-empty">No Canadian-born profiles are available in this dataset.</p>}
      </section>

      <section className="coverage-note" id="canada-re24" aria-labelledby="canada-re24-title">
        <div>
          <span className="eyebrow">Reading the gaps</span>
          <h2 id="canada-re24-title">Why RE24 can be missing.</h2>
        </div>
        <div>
          <p>RE24 comes from a separate run-value validation table. A hitter can have Selection, Contact, and Damage context without a joined RE24 value.</p>
          <p>A dash means unavailable in the current profile—not zero runs or poor performance.</p>
        </div>
      </section>

      <div className="editorial-note">
        <span className="status-dot" />
        <p>“Canadian-born” means MLB birth country equals Canada. Birthplace does not establish citizenship, nationality, or national-team eligibility.</p>
      </div>
    </div>
  );
}
