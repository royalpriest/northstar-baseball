import Link from "next/link";

const glossary = [
  {
    term: "Percentile",
    definition:
      "A league-relative rank within the available 2025 NorthStar hitter sample. A 90th percentile value means the hitter ranks above roughly 90% of profiled hitters on that measure.",
  },
  {
    term: "Selection",
    definition:
      "Swing/take decision quality after accounting for pitch and game context. It asks whether the hitter made good decisions given what was presented.",
  },
  {
    term: "Contact",
    definition:
      "Contact execution relative to how difficult the hitter's swings were expected to be. It is not the same thing as raw contact rate.",
  },
  {
    term: "Damage",
    definition:
      "Quality of contact relative to expected damage opportunity. It asks what the hitter did with the contact opportunities that actually occurred.",
  },
  {
    term: "wOBA",
    definition:
      "Weighted On-Base Average. A conventional offensive metric that gives different weights to different ways of reaching base.",
  },
  {
    term: "xwOBA",
    definition:
      "Expected wOBA based on quality-of-contact and related Statcast inputs. NorthStar displays it as an established reference metric, not as a NorthStar axis.",
  },
  {
    term: "RE24",
    definition:
      "Run expectancy change across plate appearances, including changes in base/out state and runs scored. NorthStar reports it per 100 plate appearances when the validation source has a value.",
  },
  {
    term: "Chase rate",
    definition:
      "How often a hitter swings at pitches outside the strike zone. Lower chase is generally reflected as stronger chase avoidance.",
  },
  {
    term: "Barrel rate",
    definition:
      "The share of batted balls meeting Statcast's barrel combination of exit velocity and launch angle.",
  },
];

export default function HowItWorksPage() {
  return (
    <div className="page-shell page-shell--narrow guide-page">
      <section className="page-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">Reading NorthStar</span>
          <h1 className="page-title">Three questions. No mystery grade.</h1>
          <p>
            NorthStar is designed to explain hitter shape, not compress every
            skill into one number. Start with Selection, Contact, and Damage,
            then use familiar metrics such as wOBA, xwOBA, and RE24 for context.
          </p>
        </div>
        <div className="page-hero__aside">
          2025 hitter profiles
          <br />
          Contextual diagnostics
        </div>
      </section>

      <section className="guide-steps">
        <article className="guide-step guide-step--selection">
          <span>01 / Selection</span>
          <h2>What did he choose?</h2>
          <p>
            NorthStar evaluates swing and take decisions in the situation they
            occurred. A high percentile means those decisions graded strongly
            relative to other profiled hitters after context was considered.
          </p>
        </article>
        <article className="guide-step guide-step--contact">
          <span>02 / Contact</span>
          <h2>Did he execute the swing?</h2>
          <p>
            Contact compares what happened on swings with how difficult those
            swings were expected to be. A low Contact percentile does not simply
            mean "low batting average" or "bad hitter."
          </p>
        </article>
        <article className="guide-step guide-step--damage">
          <span>03 / Damage</span>
          <h2>What did contact become?</h2>
          <p>
            Damage looks at quality of contact relative to opportunity. It is
            kept separate from Selection and Contact so very different hitter
            shapes remain visible.
          </p>
        </article>
      </section>

      <section className="guide-example">
        <span className="eyebrow">Example</span>
        <h2>How to read 98th · 4th · 100th.</h2>
        <p>
          A hitter with 98th-percentile Selection, 4th-percentile Contact, and
          100th-percentile Damage made exceptionally strong decisions and
          produced elite damage relative to context, while contact execution
          relative to swing difficulty was a much weaker part of the profile.
          NorthStar does not average those three numbers into one grade.
        </p>
        <Link href="/compare" className="link-arrow">
          See two very different hitter shapes ↗
        </Link>
      </section>

      <section className="section">
        <div className="section-header">
          <div>
            <span className="eyebrow">Glossary</span>
            <h2>Baseball terms, in plain English.</h2>
          </div>
          <p>
            These definitions explain how the product uses each term. They are
            intentionally concise rather than a replacement for the methodology.
          </p>
        </div>

        <div className="glossary-grid">
          {glossary.map((item) => (
            <article className="glossary-card" key={item.term}>
              <h3>{item.term}</h3>
              <p>{item.definition}</p>
            </article>
          ))}
        </div>
      </section>

      <div className="profile-disclaimer">
        <span className="status-dot" />
        <p>
          NorthStar's current Selection, Contact, and Damage components are
          contextual diagnostics. They are not combined into an overall score
          and are not presented as standalone novel predictive-skill claims.
        </p>
      </div>
    </div>
  );
}
