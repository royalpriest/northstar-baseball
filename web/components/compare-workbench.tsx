"use client";

import { FormEvent, useState } from "react";
import { browserApiBase } from "@/lib/browser-config";
import { decimal, percentile, signed } from "@/lib/format";
import type { CompareResponse, PlayerProfile } from "@/lib/types";

const rows: Array<{
  label: string;
  render: (player: PlayerProfile) => string;
}> = [
  { label: "Selection context", render: (p) => percentile(p.selection_context_percentile) },
  { label: "Contact context", render: (p) => percentile(p.contact_context_percentile) },
  { label: "Damage context", render: (p) => percentile(p.damage_context_percentile) },
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
  className: string;
}> = [
  {
    key: "selection_context_percentile",
    label: "Selection",
    className: "compare-axis--selection",
  },
  {
    key: "contact_context_percentile",
    label: "Contact",
    className: "compare-axis--contact",
  },
  {
    key: "damage_context_percentile",
    label: "Damage",
    className: "compare-axis--damage",
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
    <div className="compare-shell">
      <form className="compare-controls" onSubmit={submit}>
        <span className="eyebrow">Workbench</span>
        <h2>Choose 2–4 hitters</h2>
        <p>
          Use exact names or MLBAM IDs. The view keeps each metric separate —
          there is no winner or overall grade.
        </p>

        <div className="compare-input-stack">
          {players.map((player, index) => (
            <div key={index} className="compare-input-wrap">
              <span className="compare-input-index">{String(index + 1).padStart(2, "0")}</span>
              <input
                className="compare-input"
                value={player}
                onChange={(event) => update(index, event.target.value)}
                placeholder={index === 0 ? "Aaron Judge or 592450" : "Luis Arráez or 650333"}
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

        {players.length < 4 ? (
          <button type="button" onClick={addPlayer} className="compare-add">
            + Add another hitter
          </button>
        ) : null}

        <button className="primary-button" disabled={loading}>
          {loading ? "Comparing…" : "Compare profiles"}
        </button>

        {error ? <div className="notice">{error}</div> : null}
      </form>

      <section className="compare-output">
        {result ? (
          <>
            <div className="compare-output__header">
              <div>
                <span className="eyebrow">Context shape</span>
                <h2>Same scale. Different paths.</h2>
              </div>
              <span>{result.players.length} hitter profiles</span>
            </div>

            <div
              className="compare-profile-grid"
              style={{ "--compare-columns": result.players.length } as React.CSSProperties}
            >
              {result.players.map((player) => (
                <article className="compare-profile-card" key={player.batter}>
                  <div className="compare-profile-card__head">
                    <div className="compare-avatar">{initials(player.player_name)}</div>
                    <div>
                      <strong>{player.player_name}</strong>
                      <span>
                        {player.primary_team_2025 ?? "MLB"} · MLBAM {player.batter}
                      </span>
                    </div>
                  </div>

                  <div className="compare-axis-list">
                    {axes.map((axis) => {
                      const value = player[axis.key] as number | null | undefined;
                      return (
                        <div className={`compare-axis ${axis.className}`} key={axis.key}>
                          <div className="compare-axis__meta">
                            <span>{axis.label}</span>
                            <strong>{percentile(value)}</strong>
                          </div>
                          <div className="compare-axis__track">
                            <i style={{ width: `${clamped(value)}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="compare-profile-card__stats">
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
                </article>
              ))}
            </div>

            <div className="compare-matrix-header">
              <div>
                <span className="eyebrow">Metric matrix</span>
                <h3>Context beside convention.</h3>
              </div>
              <p>Percentiles are league-relative within the available 2025 sample.</p>
            </div>

            <div className="compare-table-wrap">
              <table className="compare-table">
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
                  {rows.map((row) => (
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

            <div className="compare-interpretation">
              <span className="status-dot" />
              <p>{result.interpretation}</p>
            </div>
          </>
        ) : (
          <div className="compare-empty">
            <div>
              <span className="eyebrow">Ready</span>
              <h2>Put two hitter shapes side by side.</h2>
              <p>
                Judge vs. Arráez is preloaded because it makes the three-axis
                model immediately visible: selection, contact, and damage can
                tell very different stories.
              </p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
