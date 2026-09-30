"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrandMark } from "@/components/brand-mark";

const links = [
  { href: "/players", label: "Players" },
  { href: "/blue-jays", label: "Blue Jays" },
  { href: "/canada", label: "Canada" },
  { href: "/compare", label: "Compare" },
  { href: "/how-it-works", label: "Guide" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <header className="site-nav">
      <div className="site-nav__inner">
        <Link href="/" className="brand" aria-label="NorthStar Baseball home">
          <BrandMark />
          <span>
            <strong>NorthStar</strong>
            <small>Baseball Intelligence</small>
          </span>
        </Link>

        <nav className="site-nav__links" aria-label="Primary navigation">
          {links.map((link) => {
            const active =
              pathname === link.href ||
              (link.href === "/players" && pathname.startsWith("/players/"));

            return (
              <Link
                key={link.href}
                href={link.href}
                className={`nav-link ${active ? "nav-link--active" : ""}`}
                aria-current={active ? "page" : undefined}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <div className="site-nav__status" title="2025 dataset" aria-label="2025 dataset active">
          <span className="status-dot" aria-hidden="true" />
          <span>2025</span>
        </div>
      </div>
    </header>
  );
}
