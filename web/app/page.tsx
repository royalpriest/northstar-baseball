import Link from "next/link";
import { getLeaderboard, getMeta, getTeamPlayers } from "@/lib/api";
import { SearchBox } from "@/components/search-box";
import { PlayerRow } from "@/components/player-row";
import { percentile } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [meta, canadian, toronto, damage] = await Promise.all([
    getMeta(),
    getLeaderboard("canadian", "re24_per_100_pa", 4),
    getTeamPlayers("TOR", 5),
    getLeaderboard("all", "damage_context_percentile", 3, 300),
  ]);

  return (
    <div className="page-shell">
      <section className="hero">
        <div className="hero__copy">
          <span className="kicker">Context before conclusion</span>
          <h1>
            See the hitter.
            <br />
            <span>Not just the line.</span>
          </h1>
          <p className="hero__lede">
            NorthStar separates decision quality, bat-to-ball execution, and
            damage into explainable context — then places them beside the
            established metrics you already know.
          </p>
          <SearchBox />
          <div className="hero__meta">
            <span>{meta.players} player profiles</span>
            <span>2025 MLB</span>
            <span>{meta.storage_backend === "postgresql" ? "Live PostgreSQL" : "Research snapshot"}</span>
            <span>No composite score</span>
          </div>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="orbit orbit--one">
            <span className="orbit__node orbit__node--one" />
          </div>
          <div className="orbit orbit--two">
            <span className="orbit__node orbit__node--two" />
          </div>
          <div className="orbit orbit--three">
            <span className="orbit__node orbit__node--three" />
          </div>
          <div className="signal-arc signal-arc--one" />
          <div className="signal-arc signal-arc--two" />
          <div className="signal-arc signal-arc--three" />
          <div className="hero-core">
            <div>
              <strong>3</strong>
              <span>contextual axes</span>
            </div>
          </div>
          <div className="hero-float hero-float--a">
            <span>Selection</span>
            <strong>Decision quality</strong>
          </div>
          <div className="hero-float hero-float--c">
            <span>Contact</span>
            <strong>Bat-to-ball execution</strong>
          </div>
          <div className="hero-float hero-float--b">
            <span>Damage</span>
            <strong>Quality / opportunity</strong>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-header">
          <div>
            <span className="eyebrow">NorthStar model</span>
            <h2>Three lenses. No mystery grade.</h2>
          </div>
          <p>
            Each axis answers a different question. Keeping them separate
            preserves what the research can actually support.
          </p>
        </div>

        <div className="card-grid">
          <Link href="/players" className="feature-card feature-card--selection">
            <span className="feature-card__index">01 / SELECTION</span>
            <span className="feature-card__signal" aria-hidden="true">
              <i /><i /><i />
            </span>
            <h3>What did he choose?</h3>
            <p>
              Decision value relative to count, location, pitch type, base/out
              state, and the opportunity presented.
            </p>
            <span className="feature-card__link">Explore hitter decisions ↗</span>
          </Link>
          <Link href="/players" className="feature-card feature-card--contact">
            <span className="feature-card__index">02 / CONTACT</span>
            <span className="feature-card__signal" aria-hidden="true">
              <i /><i /><i />
            </span>
            <h3>Did he make contact?</h3>
            <p>
              Contact above expectation on the swings he took, controlling for
              the difficulty of those swing opportunities.
            </p>
            <span className="feature-card__link">Explore contact context ↗</span>
          </Link>
          <Link href="/players" className="feature-card feature-card--damage">
            <span className="feature-card__index">03 / DAMAGE</span>
            <span className="feature-card__signal" aria-hidden="true">
              <i /><i /><i />
            </span>
            <h3>What did contact become?</h3>
            <p>
              xwOBA contact quality above expectation, separating impact from
              the opportunities that produced it.
            </p>
            <span className="feature-card__link">Explore damage context ↗</span>
          </Link>
        </div>
      </section>

      <section className="section">
        <div className="section-header">
          <div>
            <span className="eyebrow">Live dataset</span>
            <h2>Inside the 2025 season.</h2>
          </div>
          <Link className="link-arrow" href="/compare">
            Compare hitters ↗
          </Link>
        </div>

        <div className="split-panel">
          <div className="panel">
            <div className="panel__header">
              <h3>Canadian-born · RE24 / 100 PA</h3>
              <Link href="/canada">View all ↗</Link>
            </div>
            {canadian.items.map((player) => (
              <PlayerRow
                key={player.batter}
                player={player}
                rank={player.rank}
              />
            ))}
          </div>

          <div className="panel">
            <div className="panel__header">
              <h3>Toronto · 2025 exposure</h3>
              <Link href="/blue-jays">Blue Jays view ↗</Link>
            </div>
            {toronto.items.map((player, index) => (
              <PlayerRow
                key={player.batter}
                player={player}
                rank={index + 1}
              />
            ))}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-header">
          <div>
            <span className="eyebrow">Damage context</span>
            <h2>Impact, adjusted for opportunity.</h2>
          </div>
          <p>
            A high percentile means the hitter produced more damage than the
            modeled context would ordinarily imply.
          </p>
        </div>

        <div className="data-grid">
          {damage.items.map((player) => (
            <Link
              href={`/players/${player.batter}`}
              className="stat-panel"
              key={player.batter}
            >
              <span className="stat-panel__label">
                {player.primary_team_2025 ?? "MLB"} · Damage context
              </span>
              <div className="stat-panel__value">
                {percentile(player.damage_context_percentile)}
              </div>
              <div className="stat-panel__sub">{player.player_name}</div>
            </Link>
          ))}
        </div>
      </section>

      <footer className="footer-note">
        NorthStar contextual axes are descriptive diagnostics built from the
        validated 2025 research pipeline. They are not a composite player
        rating or standalone predictive-skill claim.
      </footer>
    </div>
  );
}
