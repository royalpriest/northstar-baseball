import Link from "next/link";
import { PlayerExplorerRow } from "@/components/player-explorer-row";
import { PlayerRow } from "@/components/player-row";
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
  {
    key: "re24",
    label: "RE24 / 100",
    sortBy: "re24_per_100_pa",
    description: "Run-value outcome",
  },
  {
    key: "selection",
    label: "Selection",
    sortBy: "selection_context_percentile",
    description: "Decision context",
  },
  {
    key: "contact",
    label: "Contact",
    sortBy: "contact_context_percentile",
    description: "Bat-to-ball context",
  },
  {
    key: "damage",
    label: "Damage",
    sortBy: "damage_context_percentile",
    description: "Quality / opportunity",
  },
];

export default async function PlayersPage({ searchParams }: Props) {
  const { q = "", lens = "re24" } = await searchParams;
  const trimmed = q.trim();
  const activeLens: Lens = lenses.some((item) => item.key === lens)
    ? (lens as Lens)
    : "re24";
  const active = lenses.find((item) => item.key === activeLens) ?? lenses[0];

  const [meta, response] = await Promise.all([
    getMeta(),
    trimmed
      ? searchPlayers(trimmed, 50)
      : getLeaderboard("all", active.sortBy, 40, 300),
  ]);

  const players = response.items;

  return (
    <div className="page-shell page-shell--narrow">
      <section className="page-hero player-index-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">Player index</span>
          <h1 className="page-title">
            {trimmed ? `Results for “${trimmed}”` : "Find a hitter."}
          </h1>
          <p>
            Search the canonical 2025 NorthStar profile by player name or
            MLBAM ID, or browse qualified hitters through one contextual lens
            at a time.
          </p>
        </div>
        <div className="page-hero__aside">
          {meta.players} hitter profiles
          <br />
          2025 regular season
        </div>
      </section>

      <div className="toolbar player-index-toolbar">
        <SearchBox compact placeholder="Search by player name or MLBAM ID…" />
      </div>

      {!trimmed ? (
        <nav className="lens-switcher" aria-label="Leaderboard lens">
          {lenses.map((item) => (
            <Link
              href={`/players?lens=${item.key}`}
              key={item.key}
              className={`lens-tab ${activeLens === item.key ? "lens-tab--active" : ""}`}
            >
              <span>{item.label}</span>
              <small>{item.description}</small>
            </Link>
          ))}
        </nav>
      ) : null}

      <section className="player-index-panel">
        <div className="player-index-panel__header">
          <div>
            <span className="eyebrow">
              {trimmed ? "Search results" : "Qualified leaderboard"}
            </span>
            <h2>{trimmed ? "Matching profiles" : active.label}</h2>
          </div>
          <div className="player-index-panel__meta">
            <strong>{players.length}</strong>
            <span>players shown</span>
          </div>
        </div>

        {!trimmed ? (
          <div className="explorer-row explorer-row--head" aria-hidden="true">
            <div>Rank</div>
            <div>Hitter</div>
            <div>NorthStar shape</div>
            <div>{active.label}</div>
            <div />
          </div>
        ) : null}

        {players.map((player, index) =>
          trimmed ? (
            <PlayerRow
              key={player.batter}
              player={player}
              rank={index + 1}
            />
          ) : (
            <PlayerExplorerRow
              key={player.batter}
              player={player}
              rank={index + 1}
              lens={activeLens}
            />
          ),
        )}
      </section>

      <div className="player-index-note">
        <span className="status-dot" />
        <p>
          Selection, Contact, and Damage remain separate contextual diagnostics.
          Switching leaderboard lenses changes the ordering, not the definition
          of the player profile.
        </p>
      </div>
    </div>
  );
}
