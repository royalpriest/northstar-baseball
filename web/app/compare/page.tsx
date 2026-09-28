import { CompareWorkbench } from "@/components/compare-workbench";

export default function ComparePage() {
  return (
    <div className="page-shell page-shell--narrow">
      <section className="page-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">Player comparison</span>
          <h1 className="page-title">Different hitters. Same scale.</h1>
          <p>
            Compare contextual NorthStar axes with established offensive
            measures. The interface is designed to reveal player shape, not
            manufacture a winner.
          </p>
        </div>
        <div className="page-hero__aside">
          2–4 hitters
          <br />
          League-relative percentiles
        </div>
      </section>

      <CompareWorkbench />
    </div>
  );
}
