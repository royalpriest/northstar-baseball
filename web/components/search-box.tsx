"use client";

import Link from "next/link";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { browserApiBase } from "@/lib/browser-config";
import type { PlayerListResponse, PlayerSummary } from "@/lib/types";

type Props = {
  compact?: boolean;
  placeholder?: string;
};

export function SearchBox({
  compact = false,
  placeholder = "Search first/last name or MLBAM ID…",
}: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PlayerSummary[]>([]);
  const [resultQuery, setResultQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
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
        if (!response.ok) {
          setResults([]);
          setResultQuery(normalized);
          return;
        }
        const data = (await response.json()) as PlayerListResponse;
        setResults(data.items);
        setResultQuery(normalized);
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          setResults([]);
          setResultQuery(normalized);
        }
      } finally {
        setLoading(false);
      }
    }, 180);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [normalized]);

  const visibleResults = resultQuery === normalized ? results : [];
  const visibleLoading =
    normalized.length >= 2 && (loading || resultQuery !== normalized);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!normalized) return;
    router.push(`/players?q=${encodeURIComponent(normalized)}`);
  }

  return (
    <div className={`search-shell ${compact ? "search-shell--compact" : ""}`}>
      <form className="search-box" onSubmit={submit} role="search">
        <span className="search-box__icon" aria-hidden="true">⌕</span>
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={placeholder}
          aria-label={placeholder}
          autoComplete="off"
          role="combobox"
          aria-expanded={normalized.length >= 2}
          aria-controls="player-search-results"
          aria-autocomplete="list"
          aria-haspopup="listbox"
        />
        <kbd>Ctrl K</kbd>
      </form>

      {normalized.length >= 2 ? (
        <div
          className="search-results"
          id="player-search-results"
          role="listbox"
          aria-label="Player search results"
        >
          <div className="search-results__meta" aria-live="polite">
            <span>{visibleLoading ? "Searching…" : "Players"}</span>
            {!visibleLoading ? <span>{visibleResults.length} shown</span> : null}
          </div>
          {visibleResults.map((player) => (
            <Link
              href={`/players/${player.batter}`}
              className="search-result"
              key={player.batter}
              role="option"
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
          {!visibleLoading && visibleResults.length === 0 ? (
            <p className="search-results__empty">No matching hitter in the 2025 profile.</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
