export default function Loading() {
  return (
    <div className="page-shell page-shell--narrow">
      <div className="page-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">NorthStar</span>
          <h1 className="page-title">Loading the board…</h1>
          <p>Reading the current 2025 profile from the NorthStar API.</p>
        </div>
      </div>

      <div className="data-grid" style={{ paddingTop: 30 }}>
        {[0, 1, 2].map((item) => (
          <div
            key={item}
            className="stat-panel"
            style={{ minHeight: 180, opacity: 0.52 }}
          />
        ))}
      </div>
    </div>
  );
}
