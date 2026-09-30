import Link from "next/link";
import { notFound } from "next/navigation";
import { LensSignature } from "@/components/lens-signature";
import { PercentileBar } from "@/components/percentile-bar";
import { getPlayer, NorthStarApiError } from "@/lib/api";
import {
  decimal,
  integer,
  percentile,
  percentRate,
  signed,
} from "@/lib/format";
import {
  percentilePlainEnglish,
  playerInterpretation,
} from "@/lib/interpretation";

export const dynamic = "force-dynamic";

type Props = {
  params: Promise<{ id: string }>;
};

export default async function PlayerPage({ params }: Props) {
  const { id } = await params;

  let player;
  try {
    player = await getPlayer(id);
  } catch (error) {
    if (error instanceof NorthStarApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  const place = [player.birth_city, player.birth_state_province, player.birth_country]
    .filter(Boolean)
    .join(", ");
  const interpretation = playerInterpretation(player);

  return (
    <div className="page-shell page-shell--narrow editorial-profile">
      <section className="editorial-profile__masthead">
        <div className="editorial-profile__identity">
          <span className="eyebrow">NorthStar hitter profile · 2025</span>
          <h1>{player.player_name}</h1>
          <div className="editorial-profile__meta">
            <span>{player.primary_team_2025 ?? "MLB"}</span>
            <span>{integer(player.season_plate_appearances)} PA</span>
            <span>MLBAM {player.batter}</span>
            {player.canadian_born ? <span>Canadian-born</span> : null}
          </div>
          {place ? <p className="editorial-profile__place">Born {place}</p> : null}
        </div>

        <div className="editorial-profile__season">
          <span>2025</span>
          <small>MLB hitter profile</small>
        </div>
      </section>

      <section className="profile-lens-stage">
        <div className="profile-lens-stage__intro">
          <span className="eyebrow">Three-lens signature</span>
          <h2>The shape of the hitter.</h2>
          <p>
            League-relative percentiles show where this player sits on each
            NorthStar lens. The midpoint marker is the 50th percentile.
          </p>
        </div>

        <LensSignature
          className="lens-signature--hero"
          selection={player.selection_context_percentile}
          contact={player.contact_context_percentile}
          damage={player.damage_context_percentile}
          selectionDetail={`${signed(player.shrunk_centered_per_100, 2)} value / 100`}
          contactDetail={`${signed(player.contact_above_expected_per_100_swings, 2)} pp / 100 swings`}
          damageDetail={`${signed(player.damage_xwoba_points_above_expected, 2)} xwOBA pts`}
        />
      </section>

      <section className="profile-reading profile-reading--editorial">
        <div className="profile-reading__header">
          <div>
            <span className="eyebrow">The read</span>
            <h2>What the profile shows.</h2>
          </div>
          <Link href="/how-it-works" className="link-arrow">
            Methodology ↗
          </Link>
        </div>

        <div className="profile-reading__grid profile-reading__grid--editorial">
          <article>
            <span>Selection · {percentile(player.selection_context_percentile)}</span>
            <p>{interpretation.selection}</p>
            <small>{percentilePlainEnglish(player.selection_context_percentile)}</small>
          </article>
          <article>
            <span>Contact · {percentile(player.contact_context_percentile)}</span>
            <p>{interpretation.contact}</p>
            <small>{percentilePlainEnglish(player.contact_context_percentile)}</small>
          </article>
          <article>
            <span>Damage · {percentile(player.damage_context_percentile)}</span>
            <p>{interpretation.damage}</p>
            <small>{percentilePlainEnglish(player.damage_context_percentile)}</small>
          </article>
        </div>
      </section>

      <section className="profile-outcome-strip">
        <div>
          <span>wOBA</span>
          <strong>{decimal(player.woba)}</strong>
          <small>{percentile(player.woba_percentile)} pct</small>
        </div>
        <div>
          <span>xwOBA</span>
          <strong>{decimal(player.est_woba)}</strong>
          <small>{percentile(player.xwoba_percentile)} pct</small>
        </div>
        <div>
          <span>RE24 / 100 PA</span>
          <strong>{signed(player.re24_per_100_pa, 2)}</strong>
          <small>{percentile(player.re24_percentile)} pct</small>
        </div>
        <div>
          <span>Season PA</span>
          <strong>{integer(player.season_plate_appearances)}</strong>
          <small>{player.primary_team_2025 ?? "MLB"} exposure</small>
        </div>
      </section>

      <section className="profile-section-intro profile-section-intro--editorial">
        <div>
          <span className="eyebrow">Under the hood</span>
          <h2>Context beside convention.</h2>
        </div>
        <p>
          The supporting measures stay visible so the contextual lenses can be
          checked against familiar outcomes rather than treated as mysterious
          scores.
        </p>
      </section>

      <section className="profile-sections profile-sections--editorial">
        <article className="detail-panel detail-panel--editorial">
          <div className="detail-panel__title">
            <span className="eyebrow">Plate discipline</span>
            <h2>What he chooses to attack.</h2>
          </div>

          <PercentileBar
            label="Chase avoidance"
            help="League-relative ranking for avoiding swings at pitches outside the strike zone."
            value={player.chase_avoidance_percentile}
            detail={`Chase rate ${percentRate(player.statcast_chase_rate)}`}
          />
          <PercentileBar
            label="Selection context"
            help="Decision quality after accounting for pitch and game context."
            value={player.selection_context_percentile}
            detail={`${integer(player.eligible_decisions)} modeled decisions`}
            tone="violet"
          />
        </article>

        <article className="detail-panel detail-panel--editorial">
          <div className="detail-panel__title">
            <span className="eyebrow">Bat-to-ball</span>
            <h2>What happens when he swings.</h2>
          </div>

          <PercentileBar
            label="Contact rate"
            help="How often the hitter made contact when swinging."
            value={player.contact_rate_percentile}
            detail={`Observed contact ${percentRate(player.statcast_contact_rate)}`}
            tone="cyan"
          />
          <PercentileBar
            label="Contact context"
            help="Contact execution relative to the modeled difficulty of the swings taken."
            value={player.contact_context_percentile}
            detail={`${integer(player.swings)} modeled swings`}
            tone="cyan"
          />
        </article>

        <article className="detail-panel detail-panel--editorial">
          <div className="detail-panel__title">
            <span className="eyebrow">Contact quality</span>
            <h2>Damage when contact is made.</h2>
          </div>

          <PercentileBar
            label="xwOBA on contact"
            help="Expected offensive value of batted-ball contact, shown here as a league-relative percentile."
            value={player.xwoba_on_contact_percentile}
            detail={decimal(player.statcast_xwoba_on_contact)}
          />
          <PercentileBar
            label="Barrel rate"
            help="Share of batted balls meeting Statcast barrel quality thresholds."
            value={player.barrel_percentile}
            detail={percentRate(player.statcast_barrel_rate)}
          />
          <PercentileBar
            label="Damage context"
            help="Contact quality relative to the opportunities presented."
            value={player.damage_context_percentile}
            detail={`${integer(player.batted_balls)} batted balls`}
            tone="violet"
          />
        </article>

        <article className="detail-panel detail-panel--editorial">
          <div className="detail-panel__title">
            <span className="eyebrow">Run production</span>
            <h2>Established outcomes.</h2>
          </div>

          <PercentileBar
            label="wOBA"
            help="Weighted On-Base Average, an established offensive outcome metric."
            value={player.woba_percentile}
            detail={decimal(player.woba)}
          />
          <PercentileBar
            label="xwOBA"
            help="Expected wOBA based on Statcast quality-of-contact inputs."
            value={player.xwoba_percentile}
            detail={decimal(player.est_woba)}
          />
          <PercentileBar
            label="RE24 / 100 PA"
            help="Run expectancy change per 100 plate appearances, when available in NorthStar's validation source."
            value={player.re24_percentile}
            detail={signed(player.re24_per_100_pa, 2)}
            tone="cyan"
          />
        </article>
      </section>

      <div className="profile-disclaimer profile-disclaimer--editorial">
        <span className="status-dot" />
        <p>
          NorthStar axes are contextual diagnostics. This page intentionally
          does not calculate a single overall player score, grade, or
          standalone predictive-skill claim.
        </p>
      </div>
    </div>
  );
}
