import { notFound } from "next/navigation";
import { PercentileBar } from "@/components/percentile-bar";
import { getPlayer } from "@/lib/api";
import {
  decimal,
  integer,
  percentile,
  percentRate,
  signed,
} from "@/lib/format";

export const dynamic = "force-dynamic";

type Props = {
  params: Promise<{ id: string }>;
};

export default async function PlayerPage({ params }: Props) {
  const { id } = await params;

  let player;
  try {
    player = await getPlayer(id);
  } catch {
    notFound();
  }

  const place = [player.birth_city, player.birth_state_province, player.birth_country]
    .filter(Boolean)
    .join(", ");

  return (
    <div className="page-shell page-shell--narrow">
      <section className="profile-header">
        <div className="profile-header__identity">
          <span className="eyebrow">NorthStar hitter profile · 2025</span>
          <h1>{player.player_name}</h1>
          <div className="profile-header__meta">
            <span className="meta-pill">{player.primary_team_2025 ?? "MLB"}</span>
            <span className="meta-pill">{integer(player.season_plate_appearances)} PA</span>
            {place ? <span className="meta-pill">{place}</span> : null}
            {player.canadian_born ? <span className="meta-pill">Canadian-born</span> : null}
          </div>
        </div>

        <div className="profile-signature" aria-label="NorthStar three-axis signature">
          <div className="profile-signature__orbit" aria-hidden="true" />
          <div className="profile-signature__center" aria-hidden="true">
            <span>
              {player.player_name
                .split(" ")
                .slice(0, 2)
                .map((part) => part[0])
                .join("")}
            </span>
          </div>
          <div className="profile-signature__axis profile-signature__axis--selection">
            <span>Selection</span>
            <strong>{percentile(player.selection_context_percentile)}</strong>
          </div>
          <div className="profile-signature__axis profile-signature__axis--contact">
            <span>Contact</span>
            <strong>{percentile(player.contact_context_percentile)}</strong>
          </div>
          <div className="profile-signature__axis profile-signature__axis--damage">
            <span>Damage</span>
            <strong>{percentile(player.damage_context_percentile)}</strong>
          </div>
          <div className="profile-id">
            <span>MLBAM</span>
            <strong>{player.batter}</strong>
          </div>
        </div>
      </section>

      <section className="axes-grid">
        <article className="axis-card axis-card--selection">
          <span className="eyebrow">Selection</span>
          <div className="axis-card__value">
            {percentile(player.selection_context_percentile)}
          </div>
          <div className="axis-card__raw">
            {signed(player.shrunk_centered_per_100, 2)} value / 100
          </div>
          <div className="axis-card__footer">
            <span>Decision quality relative to pitch and game context.</span>
            <i style={{ width: `${Math.max(0, Math.min(100, player.selection_context_percentile ?? 0))}%` }} />
          </div>
        </article>

        <article className="axis-card axis-card--contact">
          <span className="eyebrow">Contact</span>
          <div className="axis-card__value">
            {percentile(player.contact_context_percentile)}
          </div>
          <div className="axis-card__raw">
            {signed(player.contact_above_expected_per_100_swings, 2)} pp / 100 swings
          </div>
          <div className="axis-card__footer">
            <span>Contact relative to the modeled difficulty of swings taken.</span>
            <i style={{ width: `${Math.max(0, Math.min(100, player.contact_context_percentile ?? 0))}%` }} />
          </div>
        </article>

        <article className="axis-card axis-card--damage">
          <span className="eyebrow">Damage</span>
          <div className="axis-card__value">
            {percentile(player.damage_context_percentile)}
          </div>
          <div className="axis-card__raw">
            {signed(player.damage_xwoba_points_above_expected, 2)} xwOBA pts
          </div>
          <div className="axis-card__footer">
            <span>Contact quality relative to the opportunities presented.</span>
            <i style={{ width: `${Math.max(0, Math.min(100, player.damage_context_percentile ?? 0))}%` }} />
          </div>
        </article>
      </section>

      <section className="profile-section-intro">
        <div>
          <span className="eyebrow">Under the hood</span>
          <h2>Context beside convention.</h2>
        </div>
        <p>
          NorthStar keeps modeled context and established outcomes visible at
          the same time, so one number never has to carry the whole story.
        </p>
      </section>

      <section className="profile-sections">
        <article className="detail-panel">
          <span className="eyebrow">Plate discipline</span>
          <h2>What he chooses to attack.</h2>

          <PercentileBar
            label="Chase avoidance"
            value={player.chase_avoidance_percentile}
            detail={`Chase rate ${percentRate(player.statcast_chase_rate)}`}
          />
          <PercentileBar
            label="Selection context"
            value={player.selection_context_percentile}
            detail={`${integer(player.eligible_decisions)} modeled decisions`}
            tone="violet"
          />
        </article>

        <article className="detail-panel">
          <span className="eyebrow">Bat-to-ball</span>
          <h2>What happens when he swings.</h2>

          <PercentileBar
            label="Contact rate"
            value={player.contact_rate_percentile}
            detail={`Observed contact ${percentRate(player.statcast_contact_rate)}`}
            tone="cyan"
          />
          <PercentileBar
            label="Contact context"
            value={player.contact_context_percentile}
            detail={`${integer(player.swings)} modeled swings`}
            tone="cyan"
          />
        </article>

        <article className="detail-panel">
          <span className="eyebrow">Contact quality</span>
          <h2>Damage when contact is made.</h2>

          <PercentileBar
            label="xwOBA on contact"
            value={player.xwoba_on_contact_percentile}
            detail={decimal(player.statcast_xwoba_on_contact)}
          />
          <PercentileBar
            label="Barrel rate"
            value={player.barrel_percentile}
            detail={percentRate(player.statcast_barrel_rate)}
          />
          <PercentileBar
            label="Damage context"
            value={player.damage_context_percentile}
            detail={`${integer(player.batted_balls)} batted balls`}
            tone="violet"
          />
        </article>

        <article className="detail-panel">
          <span className="eyebrow">Run production</span>
          <h2>Established outcomes.</h2>

          <PercentileBar
            label="wOBA"
            value={player.woba_percentile}
            detail={decimal(player.woba)}
          />
          <PercentileBar
            label="xwOBA"
            value={player.xwoba_percentile}
            detail={decimal(player.est_woba)}
          />
          <PercentileBar
            label="RE24 / 100 PA"
            value={player.re24_percentile}
            detail={signed(player.re24_per_100_pa, 2)}
            tone="cyan"
          />
        </article>
      </section>

      <div className="profile-disclaimer">
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
