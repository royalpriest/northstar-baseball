import Link from "next/link";
import { ResponsivePlayerList } from "@/components/responsive-player-list";
import { SearchBox } from "@/components/search-box";
import { getLeaderboard, getMeta, searchPlayers } from "@/lib/api";

export const dynamic = "force-dynamic";

type Lens = "re24" | "selection" | "contact" | "damage";

type Props = {
  searchParams: Promise<{ q?: string; lens?: string }>;
};

const lenses: Array<{
  key: Lens;
  label: string;
  sortBy: string;
  description: string;
}> = [
  { key: "re24", label: "RE24 / 100", sortBy: "re24_per_100_pa", description: "Run value" },
  { key: "selection", label: "Selection", sortBy: "selection_context_percentile", description: "Decision quality" },
  { key: "contact", label: "Contact", sortBy: "contact_context_percentile", description: "Bat-to-ball" },
  { key: "damage", label: "Damage", sortBy: "damage_context_percentile", description: "Impact vs opportunity" },
];

export default async function PlayersPage({ searchParams }: Props) {
  const { q = "", lens = "re24" } = await searchParams;
  const trimmed = q.trim();
  const activeLens: Lens = lenses.some((item) => item.key === lens) ? (lens as Lens) : "re24";
  const active = lenses.find((item) => item.key === activeLens) ?? lenses[0];

  const [meta, response] = await Promise.all([
    getMeta(),
    trimmed ? searchPlayers(trimmed, 50) : getLeaderboard("all", active.sortBy, 40, 300),
  ]);

  const players = response.items;

  return (
    <div className="page-shell page-shell--narrow index-page index-page--players">
      <section className="compact-masthead">
        <div>
          <span className="eyebrow">Player index · 2025</span>
          <h1>{trimmed ? `Results for “${trimmed}”` : "Find a hitter."}</h1>
        </div>
        <div className="compact-masthead__meta">
          <strong>{meta.players}</strong>
          <span>profiles</span>
        </div>
      </section>

      <section className="explorer-toolbar">
        <SearchBox compact placeholder="Search first name, last name, or MLBAM ID…" />

        {!trimmed ? (
          <nav className="editorial-lens-switcher" aria-label="Leaderboard lens">
            {lenses.map((item) => (
              <Link
                href={`/players?lens=${item.key}`}
                key={item.key}
                className={`editorial-lens-tab ${activeLens === item.key ? "editorial-lens-tab--active" : ""}`}
              >
                <span>{item.label}</span>
                <small>{item.description}</small>
              </Link>
            ))}
          </nav>
        ) : (
          <Link href="/players" className="link-arrow explorer-toolbar__reset">Clear search ↗</Link>
        )}
      </section>

      <section className="editorial-index">
        <div className="editorial-index__heading">
          <div>
            <span className="eyebrow">{trimmed ? "Search results" : "Qualified leaderboard"}</span>
            <h2>{trimmed ? "Matching profiles" : active.label}</h2>
          </div>
          <span>{players.length} shown</span>
        </div>

        {!trimmed ? (
          <div className="explorer-row explorer-row--head editorial-row-head" aria-hidden="true">
            <div>Rank</div>
            <div>Hitter</div>
            <div>NorthStar shape</div>
            <div>{active.label}</div>
            <div />
          </div>
        ) : null}

        <ResponsivePlayerList
          players={players}
          searchMode={Boolean(trimmed)}
          lens={activeLens}
          mobileLimit={8}
        />
      </section>

      <div className="editorial-note">
        <span className="status-dot" />
        <p>Changing the lens changes the ordering only. Selection, Contact, and Damage remain separate contextual diagnostics.</p>
      </div>
    </div>
  );
}
