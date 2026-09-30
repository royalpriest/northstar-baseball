import Link from "next/link";
import { BrandMark } from "@/components/brand-mark";

const links = [
  { href: "/players", label: "Players" },
  { href: "/blue-jays", label: "Blue Jays" },
  { href: "/canada", label: "Canada" },
  { href: "/compare", label: "Compare" },
  { href: "/how-it-works", label: "Guide" },
];

export function Nav() {
  return (
    <header className="site-nav">
      <div className="site-nav__inner">
        <Link href="/" className="brand">
          <BrandMark />
          <span>
            <strong>NorthStar</strong>
            <small>Baseball Intelligence</small>
          </span>
        </Link>

        <nav className="site-nav__links" aria-label="Primary navigation">
          {links.map((link) => (
            <Link key={link.href} href={link.href} className="nav-link">
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="site-nav__status" title="2025 dataset">
          <span className="status-dot" />
          <span>2025</span>
        </div>
      </div>
    </header>
  );
}
