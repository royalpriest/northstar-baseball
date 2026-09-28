"use client";

import Link from "next/link";

export default function PlayerError({
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="page-shell page-shell--narrow">
      <div className="compare-empty" style={{ minHeight: "65vh" }}>
        <div>
          <span className="eyebrow">Profile service unavailable</span>
          <h1 className="page-title">The hitter exists. The data service is offline.</h1>
          <p style={{ color: "#91a1b5", maxWidth: 560, lineHeight: 1.7 }}>
            NorthStar could not reach the profile API. Start the FastAPI
            service, then retry this page.
          </p>
          <div style={{ display: "flex", gap: 18, justifyContent: "center", marginTop: 18 }}>
            <button className="link-arrow" onClick={() => reset()}>
              Retry ↗
            </button>
            <Link className="link-arrow" href="/players">
              Back to players ↗
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
