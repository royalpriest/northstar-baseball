import { SearchBox } from "@/components/search-box";
import { PlayerRow } from "@/components/player-row";
import { getLeaderboard, searchPlayers } from "@/lib/api";

export const dynamic = "force-dynamic";

type Props = {
  searchParams: Promise<{ q?: string }>;
};

export default async function PlayersPage({ searchParams }: Props) {
  const { q = "" } = await searchParams;
  const trimmed = q.trim();
  const response = trimmed
    ? await searchPlayers(trimmed, 50)
    : await getLeaderboard("all", "re24_per_100_pa", 40, 300);

  const players = "items" in response ? response.items : [];

  return (
    <div className="page-shell page-shell--narrow">
      <section className="page-hero">
        <div className="page-hero__copy">
          <span className="eyebrow">Player index</span>
          <h1 className="page-title">
            {trimmed ? `Results for “${trimmed}”` : "Find a hitter."}
          </h1>
          <p>
            Search the canonical 2025 NorthStar profile by player name or
            MLBAM ID. Player pages keep NorthStar context beside conventional
            measures instead of replacing them.
          </p>
        </div>
        <div className="page-hero__aside">
          2025 regular season
          <br />
          MLB hitter profiles
        </div>
      </section>

      <div className="toolbar">
        <SearchBox compact placeholder="Search by player name or MLBAM ID…" />
      </div>

      <div className="panel">
        <div className="panel__header">
          <h3>{trimmed ? "Search results" : "Qualified RE24 leaders"}</h3>
          <span>{players.length} players shown</span>
        </div>
        {players.map((player, index) => (
          <PlayerRow
            key={player.batter}
            player={player}
            rank={index + 1}
          />
        ))}
      </div>
    </div>
  );
}
