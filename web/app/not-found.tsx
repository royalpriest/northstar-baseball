import Link from "next/link";

export default function NotFound() {
  return (
    <div className="page-shell page-shell--narrow">
      <div className="compare-empty" style={{ minHeight: "65vh" }}>
        <div>
          <span className="eyebrow">404 / off the board</span>
          <h1 className="page-title">That hitter isn’t in this profile.</h1>
          <p style={{ color: "#91a1b5", maxWidth: 520, lineHeight: 1.7 }}>
            NorthStar currently serves its validated 2025 hitter dataset. Try
            searching by exact player name or MLBAM ID.
          </p>
          <Link className="link-arrow" href="/players">
            Search players ↗
          </Link>
        </div>
      </div>
    </div>
  );
}
