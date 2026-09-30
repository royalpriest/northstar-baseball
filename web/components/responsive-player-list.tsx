"use client";

import { useState } from "react";
import { PlayerExplorerRow } from "@/components/player-explorer-row";
import { PlayerRow } from "@/components/player-row";
import type { PlayerSummary } from "@/lib/types";

type Lens = "re24" | "selection" | "contact" | "damage";

type Props = {
  players: PlayerSummary[];
  searchMode: boolean;
  lens: Lens;
  mobileLimit?: number;
};

export function ResponsivePlayerList({
  players,
  searchMode,
  lens,
  mobileLimit = 8,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const hiddenCount = Math.max(0, players.length - mobileLimit);

  return (
    <div
      className={\`responsive-player-list \${expanded ? "responsive-player-list--expanded" : ""}\`}
      style={{ "--mobile-row-limit": mobileLimit } as React.CSSProperties}
    >
      <div className="responsive-player-list__rows">
        {players.map((player, index) =>
          searchMode ? (
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
              lens={lens}
            />
          ),
        )}
      </div>

      {hiddenCount > 0 ? (
        <button
          type="button"
          className="mobile-show-more"
          onClick={() => setExpanded((current) => !current)}
          aria-expanded={expanded}
        >
          <span>{expanded ? "Show less" : "Show more"}</span>
          <small>
            {expanded ? \`\${players.length} shown\` : \`\${mobileLimit} of \${players.length} shown\`}
          </small>
        </button>
      ) : null}
    </div>
  );
}
