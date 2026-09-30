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
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <div className="ambient ambient--one" aria-hidden="true" />
        <div className="ambient ambient--two" aria-hidden="true" />
        <Nav />
        <main id="main-content">{children}</main>
      </body>
    </html>
  );
}
