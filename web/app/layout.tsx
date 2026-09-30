import type { Metadata } from "next";
import "@/app/globals.css";
import { Nav } from "@/components/nav";

export const metadata: Metadata = {
  title: {
    default: "NorthStar Baseball",
    template: "%s · NorthStar Baseball",
  },
  description:
    "Context-aware MLB hitter analysis built from NorthStar's validated 2025 research pipeline.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <div className="ambient ambient--one" aria-hidden="true" />
        <div className="ambient ambient--two" aria-hidden="true" />
        <Nav />
        <main id="main-content" tabIndex={-1}>{children}</main>
        <footer className="site-footer">
          <div className="site-footer__inner">
            <span className="site-footer__brand">NorthStar Baseball</span>
            <nav className="site-footer__links" aria-label="Creator links">
              <a
                href="https://royalpriest.dev"
                target="_blank"
                rel="noreferrer"
              >
                Portfolio ↗
              </a>
              <a
                href="https://github.com/royalpriest"
                target="_blank"
                rel="noreferrer"
              >
                GitHub ↗
              </a>
            </nav>
          </div>
        </footer>
      </body>
    </html>
  );
}
