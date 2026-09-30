import Link from "next/link";

const glossary = [
  ["Percentile", "League-relative rank within the available 2025 NorthStar hitter sample."],
  ["Selection", "Swing/take decision quality after accounting for pitch and game context."],
  ["Contact", "Contact execution relative to the modeled difficulty of the swings taken."],
  ["Damage", "Quality of contact relative to expected damage opportunity."],
  ["wOBA", "Weighted On-Base Average, shown as an established offensive reference metric."],
  ["xwOBA", "Expected wOBA based on Statcast quality-of-contact inputs."],
  ["RE24", "Run expectancy change across plate appearances, reported per 100 PA when available."],
  ["Chase rate", "How often a hitter swings at pitches outside the strike zone."],
  ["Barrel rate", "Share of batted balls meeting Statcast barrel exit-velocity and launch-angle thresholds."],
] as const;

export default function HowItWorksPage() {
  return (
    <div className="page-shell page-shell--narrow guide-page guide-page--editorial">
      <section className="compact-masthead compact-masthead--split">
        <div>
          <span className="eyebrow">Reading NorthStar</span>
          <h1>Three questions. No mystery grade.</h1>
        </div>
        <p className="compact-masthead__lede">Start with Selection, Contact, and Damage. Then use familiar outcomes such as wOBA, xwOBA, and RE24 to check the shape against conventional context.</p>
      </section>

      <section className="guide-sequence">
        <article className="guide-sequence__row guide-sequence__row--selection">
          <span>01</span>
          <div><strong>Selection</strong><small>Decision quality</small></div>
          <h2>What did he choose?</h2>
          <p>Evaluates swing and take decisions in the context in which they occurred.</p>
        </article>
        <article className="guide-sequence__row guide-sequence__row--contact">
          <span>02</span>
          <div><strong>Contact</strong><small>Bat-to-ball</small></div>
          <h2>Did he execute the swing?</h2>
          <p>Compares actual contact with how difficult those swings were expected to be.</p>
        </article>
        <article className="guide-sequence__row guide-sequence__row--damage">
          <span>03</span>
          <div><strong>Damage</strong><small>Impact</small></div>
          <h2>What did contact become?</h2>
          <p>Measures contact quality relative to the opportunities that produced it.</p>
        </article>
      </section>

      <section className="guide-example guide-example--editorial">
        <div>
          <span className="eyebrow">Example profile</span>
          <h2>98 · 4 · 100</h2>
        </div>
        <div>
          <p><strong>98 Selection</strong> — exceptionally strong contextual decisions.</p>
          <p><strong>4 Contact</strong> — weak contact execution relative to swing difficulty.</p>
          <p><strong>100 Damage</strong> — elite contact quality relative to opportunity.</p>
          <Link href="/compare" className="link-arrow">Compare hitter shapes ↗</Link>
        </div>
      </section>

      <section className="glossary-sheet">
        <div className="glossary-sheet__heading">
          <span className="eyebrow">Glossary</span>
          <h2>The terms NorthStar uses.</h2>
        </div>
        <div className="glossary-sheet__list">
          {glossary.map(([term, definition]) => (
            <div className="glossary-sheet__row" key={term}>
              <strong>{term}</strong>
              <p>{definition}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="editorial-note">
        <span className="status-dot" />
        <p>Selection, Contact, and Damage are contextual diagnostics. They are not combined into an overall score or presented as standalone predictive-skill claims.</p>
      </div>
    </div>
  );
}
