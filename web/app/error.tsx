"use client";

import Link from "next/link";

export default function AppError({
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="page-shell page-shell--narrow">
      <div className="compare-empty" style={{ minHeight: "65vh" }}>
        <div>
          <span className="eyebrow">NorthStar hit a snag</span>
          <h1 className="page-title">This view could not be loaded.</h1>
          <p style={{ color: "#91a1b5", maxWidth: 560, lineHeight: 1.7 }}>
            The data service may be temporarily unavailable. Retry the request,
            or return home and continue from another view.
          </p>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 18,
              justifyContent: "center",
              marginTop: 18,
            }}
          >
            <button className="link-arrow" type="button" onClick={() => reset()}>
              Retry ↗
            </button>
            <Link className="link-arrow" href="/">
              Back home ↗
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
