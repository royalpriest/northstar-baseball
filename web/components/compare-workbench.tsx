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

        {players.map((player, index) => (
          <div key={index} style={{ position: "relative" }}>
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
                aria-label={`Remove player ${index + 1}`}
                style={{
                  position: "absolute",
                  right: 10,
                  top: 11,
                  border: 0,
                  background: "transparent",
                  color: "#65758a",
                  cursor: "pointer",
                }}
              >
                ×
              </button>
            ) : null}
          </div>
        ))}

        {players.length < 4 ? (
          <button
            type="button"
            onClick={addPlayer}
            className="filter-pill"
            style={{ width: "100%", marginBottom: 10 }}
          >
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
            <table className="compare-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  {result.players.map((player) => (
                    <th key={player.batter}>
                      {player.player_name}
                      <br />
                      <span style={{ color: "#65758a", fontWeight: 500 }}>
                        {player.primary_team_2025 ?? "MLB"}
                      </span>
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
            <div className="notice" style={{ margin: 18 }}>
              {result.interpretation}
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
