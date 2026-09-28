"use client";

import Link from "next/link";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { browserApiBase } from "@/lib/browser-config";
import type { PlayerListResponse, PlayerSummary } from "@/lib/types";

type Props = {
  compact?: boolean;
  placeholder?: string;
};

export function SearchBox({
  compact = false,
  placeholder = "Search any MLB hitter…",
}: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PlayerSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const normalized = useMemo(() => query.trim(), [query]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (normalized.length < 2) {
      setResults([]);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `${browserApiBase}/api/v1/players?query=${encodeURIComponent(normalized)}&limit=6`,
          { signal: controller.signal },
        );
        if (!response.ok) return;
        const data = (await response.json()) as PlayerListResponse;
        setResults(data.items);
      } catch (error) {
        if ((error as Error).name !== "AbortError") setResults([]);
      } finally {
        setLoading(false);
      }
    }, 180);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [normalized]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!normalized) return;
    window.location.href = `/players?q=${encodeURIComponent(normalized)}`;
  }

  return (
    <div className={`search-shell ${compact ? "search-shell--compact" : ""}`}>
      <form className="search-box" onSubmit={submit}>
        <span className="search-box__icon" aria-hidden="true">⌕</span>
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={placeholder}
          aria-label={placeholder}
          autoComplete="off"
        />
        <kbd>Ctrl K</kbd>
      </form>

      {normalized.length >= 2 ? (
        <div className="search-results">
          <div className="search-results__meta">
            <span>{loading ? "Searching…" : "Players"}</span>
            {!loading ? <span>{results.length} shown</span> : null}
          </div>
          {results.map((player) => (
            <Link
              href={`/players/${player.batter}`}
              className="search-result"
              key={player.batter}
            >
              <span className="search-result__initials">
                {player.player_name
                  .split(" ")
                  .slice(0, 2)
                  .map((part) => part[0])
                  .join("")}
              </span>
              <span className="search-result__copy">
                <strong>{player.player_name}</strong>
                <small>
                  {player.primary_team_2025 ?? "MLB"} · MLBAM {player.batter}
                </small>
              </span>
              <span>↗</span>
            </Link>
          ))}
          {!loading && results.length === 0 ? (
            <p className="search-results__empty">No matching hitter in the 2025 profile.</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
