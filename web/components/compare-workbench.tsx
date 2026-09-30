"use client";

import { FormEvent, useState, type CSSProperties } from "react";
import { browserApiBase } from "@/lib/browser-config";
import { decimal, percentile, signed } from "@/lib/format";
import type { CompareResponse, PlayerProfile } from "@/lib/types";

const supportRows: Array<{
  label: string;
  render: (player: PlayerProfile) => string;
}> = [
  { label: "Chase avoidance", render: (p) => percentile(p.chase_avoidance_percentile) },
  { label: "Contact rate", render: (p) => percentile(p.contact_rate_percentile) },
  { label: "xwOBA on contact", render: (p) => percentile(p.xwoba_on_contact_percentile) },
  { label: "Barrel rate", render: (p) => percentile(p.barrel_percentile) },
  { label: "wOBA", render: (p) => decimal(p.woba) },
  { label: "xwOBA", render: (p) => decimal(p.est_woba) },
  { label: "RE24 / 100 PA", render: (p) => signed(p.re24_per_100_pa, 2) },
];

const axes: Array<{
  key: "selection_context_percentile" | "contact_context_percentile" | "damage_context_percentile";
  label: string;
  note: string;
  className: string;
}> = [
  {
    key: "selection_context_percentile",
    label: "Selection",
    note: "Decision quality",
    className: "selection",
  },
  {
    key: "contact_context_percentile",
    label: "Contact",
    note: "Bat-to-ball execution",
    className: "contact",
  },
  {
    key: "damage_context_percentile",
    label: "Damage",
    note: "Contact quality vs opportunity",
    className: "damage",
  },
];

function clamped(value?: number | null) {
  return Math.max(0, Math.min(100, value ?? 0));
}

function initials(name: string) {
  return name
    .split(" ")
    .slice(0, 2)
    .map((part) => part[0])
    .join("");
}

export function CompareWorkbench() {
  const [players, setPlayers] = useState(["592450", "650333"]);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function update(index: number, value: string) {
    setPlayers((current) =>
      current.map((player, playerIndex) => (playerIndex === index ? value : player)),
    );
  }

  function addPlayer() {
    if (players.length < 4) setPlayers((current) => [...current, ""]);
  }

  function removePlayer(index: number) {
    if (players.length <= 2) return;
    setPlayers((current) => current.filter((_, playerIndex) => playerIndex !== index));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const selected = players.map((player) => player.trim()).filter(Boolean);

    if (selected.length < 2) {
      setError("Add at least two player names or MLBAM IDs.");
      return;
    }

    setLoading(true);
    setError("");

    const query = selected
      .map((player) => `player=${encodeURIComponent(player)}`)
      .join("&");

    try {
      const response = await fetch(`${browserApiBase}/api/v1/compare?${query}`);
      const data = (await response.json()) as CompareResponse | { detail?: string };
      if (!response.ok || !("players" in data)) {
        throw new Error("detail" in data && data.detail ? data.detail : "Comparison failed.");
      }
      setResult(data);
    } catch (requestError) {
      setResult(null);
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="compare-shell compare-shell--editorial">
      <form className="compare-controls compare-controls--editorial" onSubmit={submit}>
        <div className="compare-controls__intro">
          <span className="eyebrow">Set the field</span>
          <h2>Choose 2–4 hitters.</h2>
          <p>
            Use a first name, last name, full name, or MLBAM ID.
          </p>
        </div>

        <div className="compare-input-stack compare-input-stack--editorial">
          {players.map((player, index) => (
            <div key={index} className="compare-input-wrap compare-input-wrap--editorial">
              <span className="compare-input-index">{String(index + 1).padStart(2, "0")}</span>
              <input
                className="compare-input"
                value={player}
                onChange={(event) => update(index, event.target.value)}
                placeholder={index === 0 ? "Judge, Aaron, or 592450" : "Arráez, Luis, or 650333"}
                aria-label={`Player ${index + 1}`}
              />
              {players.length > 2 ? (
                <button
                  type="button"
                  onClick={() => removePlayer(index)}
                  className="compare-remove"
                  aria-label={`Remove player ${index + 1}`}
                >
                  ×
                </button>
              ) : null}
            </div>
          ))}
        </div>

        <div className="compare-controls__actions">
          {players.length < 4 ? (
            <button type="button" onClick={addPlayer} className="compare-add">
              + Add hitter
            </button>
          ) : <span />}
          <button className="primary-button" disabled={loading}>
            {loading ? "Comparing…" : "Compare"}
          </button>
        </div>

        {error ? <div className="notice" role="alert">{error}</div> : null}
      </form>

      <section className="compare-output compare-output--editorial" aria-live="polite" aria-busy={loading}>
        {result ? (
          <>
            <div className="compare-output__header compare-output__header--editorial">
              <div>
                <span className="eyebrow">Comparison board</span>
                <h2>Same scale. Clear differences.</h2>
              </div>
              <span>2025 · league-relative percentiles</span>
            </div>

            <div
              className="comparison-board"
              style={{ "--compare-columns": result.players.length } as CSSProperties}
            >
              <div className="comparison-board__corner">
                <span>NorthStar lenses</span>
                <small>50 = league midpoint</small>
              </div>

              {result.players.map((player, index) => (
                <div className="comparison-board__player" key={player.batter}>
                  <div className={`comparison-board__badge comparison-board__badge--${index + 1}`}>
                    {initials(player.player_name)}
                  </div>
                  <div>
                    <strong>{player.player_name}</strong>
                    <span>{player.primary_team_2025 ?? "MLB"} · {player.season_plate_appearances ?? "—"} PA</span>
                  </div>
                </div>
              ))}

              {axes.map((axis) => (
                <div className={`comparison-board__metric-row comparison-board__metric-row--${axis.className}`} key={axis.key}>
                  <div className="comparison-board__metric-label">
                    <strong>{axis.label}</strong>
                    <span>{axis.note}</span>
                  </div>

                  {result.players.map((player, index) => {
                    const value = player[axis.key] as number | null | undefined;
                    const position = clamped(value);
                    return (
                      <div className="comparison-board__cell" key={player.batter}>
                        <div className="comparison-board__value">
                          <strong>{percentile(value)}</strong>
                          <span>pct</span>
                        </div>
                        <div className="comparison-board__track" aria-label={`${player.player_name} ${axis.label}: ${percentile(value)} percentile`}>
                          <span className="comparison-board__midpoint" />
                          <span
                            className={`comparison-board__fill comparison-board__fill--player-${index + 1}`}
                            style={{ width: `${position}%` }}
                          />
                          <span
                            className={`comparison-board__marker comparison-board__marker--player-${index + 1}`}
                            style={{ left: `${position}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            <div
              className="compare-outcomes"
              style={{ "--compare-columns": result.players.length } as CSSProperties}
            >
              <div className="compare-outcomes__label">
                <span>Established outcomes</span>
              </div>
              {result.players.map((player) => (
                <div className="compare-outcomes__player" key={player.batter}>
                  <div>
                    <span>wOBA</span>
                    <strong>{decimal(player.woba)}</strong>
                  </div>
                  <div>
                    <span>xwOBA</span>
                    <strong>{decimal(player.est_woba)}</strong>
                  </div>
                  <div>
                    <span>RE24 / 100</span>
                    <strong>{signed(player.re24_per_100_pa, 2)}</strong>
                  </div>
                </div>
              ))}
            </div>

            <div className="compare-support">
              <div className="compare-support__heading">
                <div>
                  <span className="eyebrow">Supporting measures</span>
                  <h3>More context, without another wall of cards.</h3>
                </div>
                <p>All percentile rows use the same 2025 league-relative frame.</p>
              </div>

              <div className="compare-table-wrap compare-table-wrap--editorial">
                <table className="compare-table">
                  <caption className="sr-only">
                    Supporting offensive metrics for the selected hitters
                  </caption>
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {result.players.map((player) => (
                        <th key={player.batter}>
                          {player.player_name}
                          <br />
                          <span>{player.primary_team_2025 ?? "MLB"}</span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {supportRows.map((row) => (
                      <tr key={row.label}>
                        <td>{row.label}</td>
                        {result.players.map((player) => (
                          <td key={player.batter}>{row.render(player)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="compare-interpretation compare-interpretation--editorial">
              <span className="status-dot" />
              <p>{result.interpretation}</p>
            </div>
          </>
        ) : (
          <div className="compare-empty compare-empty--editorial compare-empty--compact">
            <span className="eyebrow">Ready</span>
            <h2>Judge and Arráez are preloaded.</h2>
            <p>
              Compare them to see how very different combinations of selection,
              contact, and damage can sit on the same league-relative scale.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
