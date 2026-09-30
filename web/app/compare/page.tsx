import { CompareWorkbench } from "@/components/compare-workbench";

export default function ComparePage() {
  return (
    <div className="page-shell page-shell--narrow compare-page--editorial">
      <section className="compare-masthead">
        <span className="eyebrow">Player comparison · 2025</span>
        <div className="compare-masthead__grid">
          <h1>
            Different hitters.
            <br />
            <em>Same scale.</em>
          </h1>
          <p>
            Put 2–4 hitter profiles on one league-relative field. NorthStar
            shows where their shapes diverge without turning the comparison
            into a winner or a composite ranking.
          </p>
        </div>
        <div className="compare-masthead__rule">
          <span>0</span>
          <span>League percentile</span>
          <span>100</span>
        </div>
      </section>

      <CompareWorkbench />
    </div>
  );
}
