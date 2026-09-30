import Link from "next/link";
import { LensSignature } from "@/components/lens-signature";
import { PlayerRow } from "@/components/player-row";
import { SearchBox } from "@/components/search-box";
import { getLeaderboard, getMeta, getTeamPlayers } from "@/lib/api";
import { percentile } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [meta, canadian, toronto, damage] = await Promise.all([
    getMeta(),
    getLeaderboard("canadian", "re24_per_100_pa", 4),
    getTeamPlayers("TOR", 5),
    getLeaderboard("all", "damage_context_percentile", 4, 300),
  ]);

  const featured = damage.items[0];

  return (
    <div className="page-shell editorial-home">
      <section className="editorial-hero">
        <div className="editorial-hero__intro">
          <span className="kicker">NorthStar · 2025 MLB</span>
          <h1>
            A different way
            <br />
            <em>to read hitting.</em>
          </h1>
          <p>
            Separate what a hitter chose, what he connected with, and what that
            contact became. NorthStar keeps those three questions independent
            instead of collapsing them into one grade.
          </p>
          <SearchBox placeholder="Search Judge, Arráez, Guerrero, or MLBAM ID…" />
          <div className="editorial-hero__notes">
            <span>{meta.players} hitters</span>
            <span>League-relative context</span>
            <span>{meta.storage_backend === "postgresql" ? "PostgreSQL" : "Research snapshot"}</span>
            <span>No composite score</span>
          </div>
        </div>

        {featured ? (
          <Link href={`/players/${featured.batter}`} className="featured-profile">
            <div className="featured-profile__topline">
              <span>Featured profile</span>
              <span>{featured.primary_team_2025 ?? "MLB"} · 2025</span>
            </div>
            <div className="featured-profile__identity">
              <div>
                <strong>{featured.player_name}</strong>
                <span>Three lenses, one hitter shape</span>
              </div>
              <span className="featured-profile__arrow">↗</span>
            </div>
            <LensSignature
              selection={featured.selection_context_percentile}
              contact={featured.contact_context_percentile}
              damage={featured.damage_context_percentile}
            />
            <div className="featured-profile__footer">
              <span>Damage context</span>
              <strong>{percentile(featured.damage_context_percentile)}</strong>
            </div>
          </Link>
        ) : null}
      </section>

      <section className="editorial-section editorial-section--model">
        <div className="editorial-section__intro">
          <span className="eyebrow">The model</span>
          <h2>Three questions. Kept separate on purpose.</h2>
          <p>
            Each lens measures a different part of the plate appearance. The
            separation is the point: strong damage does not erase weak contact,
            and elite contact does not automatically imply elite decisions.
          </p>
        </div>

        <div className="lens-definition-list">
          <Link href="/players" className="lens-definition lens-definition--selection">
            <span className="lens-definition__index">01</span>
            <div>
              <strong>Selection</strong>
              <p>Was the swing or take valuable given pitch and game context?</p>
            </div>
            <span className="lens-definition__mark">Decision quality ↗</span>
          </Link>
          <Link href="/players" className="lens-definition lens-definition--contact">
            <span className="lens-definition__index">02</span>
            <div>
              <strong>Contact</strong>
              <p>Did the hitter make contact relative to the difficulty of the swings he took?</p>
            </div>
            <span className="lens-definition__mark">Bat-to-ball ↗</span>
          </Link>
          <Link href="/players" className="lens-definition lens-definition--damage">
            <span className="lens-definition__index">03</span>
            <div>
              <strong>Damage</strong>
              <p>How much contact quality did the hitter create relative to opportunity?</p>
            </div>
            <span className="lens-definition__mark">Impact ↗</span>
          </Link>
        </div>
      </section>

      <section className="editorial-section">
        <div className="editorial-section__heading">
          <div>
            <span className="eyebrow">Inside the season</span>
            <h2>Read the dataset, not a leaderboard alone.</h2>
          </div>
          <Link href="/compare" className="link-arrow">
            Open comparison ↗
          </Link>
        </div>

        <div className="editorial-duo">
          <article className="editorial-list">
            <div className="editorial-list__head">
              <div>
                <span>Canadian-born</span>
                <strong>RE24 / 100 PA</strong>
              </div>
              <Link href="/canada">Canada view ↗</Link>
            </div>
            {canadian.items.map((player) => (
              <PlayerRow key={player.batter} player={player} rank={player.rank} />
            ))}
          </article>

          <article className="editorial-list">
            <div className="editorial-list__head">
              <div>
                <span>Toronto</span>
                <strong>2025 exposure</strong>
              </div>
              <Link href="/blue-jays">Blue Jays view ↗</Link>
            </div>
            {toronto.items.map((player, index) => (
              <PlayerRow key={player.batter} player={player} rank={index + 1} />
            ))}
          </article>
        </div>
      </section>

      <section className="editorial-section editorial-section--leaders">
        <div className="editorial-section__intro editorial-section__intro--wide">
          <span className="eyebrow">Damage context</span>
          <h2>Impact after accounting for the opportunity.</h2>
          <p>
            These are not “best hitter” rankings. They show where contact
            quality most exceeded NorthStar&apos;s modeled expectation in the
            available 2025 sample.
          </p>
        </div>

        <div className="editorial-rank">
          {damage.items.map((player, index) => (
            <Link href={`/players/${player.batter}`} className="editorial-rank__row" key={player.batter}>
              <span className="editorial-rank__number">{String(index + 1).padStart(2, "0")}</span>
              <div className="editorial-rank__identity">
                <strong>{player.player_name}</strong>
                <span>{player.primary_team_2025 ?? "MLB"}</span>
              </div>
              <LensSignature
                compact
                selection={player.selection_context_percentile}
                contact={player.contact_context_percentile}
                damage={player.damage_context_percentile}
              />
              <div className="editorial-rank__metric">
                <strong>{percentile(player.damage_context_percentile)}</strong>
                <span>Damage</span>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <footer className="footer-note editorial-footer">
        NorthStar contextual axes are descriptive diagnostics from the validated
        2025 research pipeline. They are not a composite player rating or a
        standalone predictive-skill claim.
      </footer>
    </div>
  );
}
