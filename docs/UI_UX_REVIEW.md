# NorthStar UI/UX — full-site review

Date: 2026-10-29
Working branch: `claude/trusting-gates-upm95z`
Scope: every route in `web/app`, every component in `web/components`, and `web/app/globals.css`.

This review applies the same standard used for Media Vault (Storikeep) UI work: one responsive owner per
rule, 44 × 44px touch targets, no horizontal clipping at any supported width, real semantics for assistive
technology, readable text at 200% zoom, regression tests that lock the invariants in, and explicit user
sign-off before merging.

**Method and limits.** This is a source review. The site was not rendered for this pass, so every layout
finding below that depends on real pixel measurements is marked *(verify rendered)* and must be confirmed in
a browser before it is fixed. Code-level findings (semantics, data display, CSS structure, contrast of
declared colours) are confirmed from the source.

## Summary

| Area | Status | Headline |
| --- | --- | --- |
| Data correctness in UI | **Fix first** | Missing percentiles render as `0` on Players and Blue Jays rows, contradicting the "a dash means unavailable" rule the Canada page states. |
| CSS structure | **High** | 7,136 lines from at least four stacked design iterations; ~125 of 390 classes are unused; 332 selectors are declared more than once; 14 different breakpoints. |
| Touch targets | **High** | Mobile dock links are 40–42px tall; the Compare "×" remove button has no size at all. |
| Readability | **High** | 134 `font-size` declarations are under 12px (down to 7px); two muted text colours fail WCAG AA contrast. |
| Search | **High** | Announced as a combobox but has no arrow-key, Escape, or outside-click behaviour. |
| Semantics | Medium | Leaderboards are not lists, column headers are hidden from screen readers, `aria-label` is used on plain `div`s where it is ignored. |
| Consistency | Medium | Loading and error screens use the retired visual system, inline styles and developer-facing copy. |
| Tests | Medium | No layout or accessibility regression tests exist for the web app. |

## 1. Data shown incorrectly (fix first)

- **Missing values shown as zero.** `components/player-explorer-row.tsx:61,66,71` and
  `components/team-context-row.tsx:42,47,52` render `Math.round(value ?? 0)`, so a hitter with no Selection,
  Contact or Damage value shows **0** and an empty bar. The Canada page (`app/canada/page.tsx`) explicitly
  tells readers "A dash means unavailable, not zero", and `components/canada-player-row.tsx` already does this
  correctly with `hasValue()` and "—". Adopt the Canada row behaviour everywhere.
- **Home lens cards all link to the same page.** `app/page.tsx:82,90,98` — the Selection, Contact and Damage
  cards all go to `/players`. They should open `/players?lens=selection|contact|damage`.
- **Unformatted PA on Compare.** `components/compare-workbench.tsx:188` prints the raw number; every other
  page uses `integer()` (e.g. `1,024`).

## 2. CSS structure

`globals.css` carries comment-marked layers ("Site-wide consistency, mobile, and accessibility pass",
"NorthStar Editorial Data System", "signature typography + flat compact home", …) that each re-declare the
earlier ones instead of replacing them. This is the same problem the Media Vault list cleanup fixed.

- **Dead code:** about 125 classes are not referenced anywhere in the app (e.g. the whole `hero*`, `orbit*`,
  `signal-arc*`, `axis-card*`, `compare-spectrum*`, `compare-profile-card*`, `toronto-leader-card*`,
  `guide-step*`, `glossary-card` families).
- **Repeated owners:** `.editorial-home` is declared 11 times; `.nav-link`, `.site-nav__links` and
  `.site-nav__inner` 7 times each; `body` 4 times; `:root` 3 times (`--muted-2` is defined twice with
  different values).
- **Breakpoint sprawl:** 14 different `max-width` values (1220, 1080, 1050, 1024, 980, 900, 820, 760, 700,
  640, 620, 560, 430). Several are within 20–40px of each other, which is where overlapping rules and
  specificity mismatches hide.
- **Tokens bypassed:** 115 distinct hex colours and 289 `rgba()` literals versus 324 `var(--…)` uses;
  three error screens hard-code `#91a1b5` inline.
- **Overrides:** 20 `!important` declarations (most are the standard `.sr-only` utility; the rest should be
  justified or removed).

**Proposed breakpoint set** (one owner per range, mobile-first where practical):

| Range | Purpose |
| --- | --- |
| ≤380px | Smallest phones: icon-led actions, tightest gaps |
| 381–520px | Standard phones |
| 521–767px | Large phones / small tablets in portrait |
| 768–1199px | Tablets and small laptops (the bottom dock currently switches at 820px — decide whether it stays) |
| ≥1200px | Desktop |

## 3. Touch targets (44 × 44px minimum)

| Control | Current | Location |
| --- | --- | --- |
| Mobile dock nav links | 42px (≤820px), 40px (≤430px) tall | `globals.css` ~3921, ~4003 |
| Compare "Remove player" × | No width/height; glyph-sized | `.compare-remove`, `globals.css` ~2205 |
| Metric help term (`?`) | Inline text-sized focus target | `components/metric-term.tsx` |
| Search result rows, lens tabs | Likely ≥44px *(verify rendered)* | `.search-result`, `.editorial-lens-tab` |
| Footer links | *(verify rendered)* | `.site-footer__links a` |

Already compliant: "Show more" (52px), "+ Add hitter" (46px).

## 4. Readability and contrast

- **Tiny type:** 134 declarations below 12px, including 7px, 7.5px, 8px (16×) and 9px (24×). Space Mono at
  these sizes is hard to read and degrades badly on low-DPI screens. Proposed floor: 12px for any text a
  reader must understand, 11px only for decorative labels.
- **Dock labels** are 10px / 9.5px uppercase in five equal columns at 320px width *(verify rendered for
  clipping of "Blue Jays")*.
- **Contrast against `--bg` (#07111d), WCAG AA needs 4.5:1 for body text:**

  | Colour | Ratio | Result |
  | --- | --- | --- |
  | `#586d85` (lens-tab `small`) | 3.56 | Fails |
  | `#65758a` (old `--muted-2`, `.compare-remove`) | 4.03 | Fails |
  | `#74869c` (current `--muted-2`) | 5.09 | Passes |
  | `#91a1b5` (`--muted`) | 7.20 | Passes |

  Panels sit on lighter translucent fills, so effective ratios are lower *(verify rendered)*.
- **Tooltips:** `.metric-term::after` text is 10px, only appears on hover/focus, cannot be dismissed with
  Escape (WCAG 1.4.13), and is not reachable by touch users in a dependable way.

## 5. Search (`components/search-box.tsx`)

- Declares `role="combobox"` / `listbox` / `option` but implements none of the keyboard model: no
  ArrowUp/ArrowDown, no `aria-activedescendant`, no Escape to close, no Enter on a highlighted result.
- Results stay open after focus leaves the field and there is no outside-click close; on mobile the panel is
  `position: fixed` and can cover the page.
- The listbox contains non-option children (the live-region meta row and the empty message).
- `aria-label` repeats the long placeholder; use a short label such as "Search players".
- The `Ctrl K` hint is hidden on phones but shows "Ctrl" to Mac users; show `⌘K` on Apple platforms.

## 6. Semantics and screen-reader output

- **Leaderboards are not lists.** Ranked rows on Home, Players, Blue Jays and Canada are bare links inside
  `div`s. Use `<ol>` so position and count are announced.
- **Column headers hidden.** `app/players/page.tsx:83` and `app/blue-jays/page.tsx:81` mark the header row
  `aria-hidden`, so a row reads as "01 Name TOR 600 PA S 80 C 45 D 72 +1.20 ↗" with no labels. Give each
  value a visually hidden label, or move to a real table.
- **Ignored labels.** `aria-label` on plain `div`s (`lens-signature.tsx:59`, `player-explorer-row.tsx:57`,
  `compare-workbench.tsx:209`, `nav.tsx:48`) is not announced. Use visible text, `sr-only` text, or
  `role="img"` with a label where appropriate.
- **Decorative glyphs announced.** "↗" inside link text is read as "north east arrow" on most pages; the
  Canada row already hides it with `aria-hidden`. Apply that everywhere.
- **Active lens tab** (`app/players/page.tsx:61`) has no `aria-current`.
- **`MetricTerm`** is a focusable `span` with no role; its `aria-label` replaces the visible label.
- **Compare form errors** are not tied to the inputs (`aria-invalid`, `aria-describedby`).

## 7. Page-level UX

- **Compare:** the result is not in the URL, so a comparison cannot be shared, bookmarked or restored after
  reload. Inputs require exact names or IDs and do not reuse the existing player search suggestions. The
  "Judge and Arráez are preloaded" empty state still requires a click — run it automatically, or say
  "Press Compare". With 4 players the board needs 810px and scrolls sideways on phones without a sticky label
  column *(verify rendered)*.
- **Navigation naming:** the nav says "Guide" but the page title is "Three questions. No mystery grade." at
  `/how-it-works`. Use one name.
- **Loading screen** (`app/loading.tsx`) uses `page-hero`, `data-grid`, `stat-panel` from the retired design,
  so every navigation flashes a different visual style. It also lacks `role="status"` / `aria-busy`.
- **Error screens** (`app/error.tsx`, `app/not-found.tsx`, `app/players/[id]/error.tsx`) use inline styles,
  and the player error tells visitors to "Start the FastAPI service" — developer copy in production.
- **Inclusive copy:** player copy and the guide assume "he" ("What did he choose?"). Use "the hitter".

## 8. What already works

- Skip link, `main` focus target, and a global `:focus-visible` ring.
- `prefers-reduced-motion` handling.
- Mobile "Show more" pattern with `aria-expanded` and a 52px target.
- Canada rows: dash-for-missing, hidden decorative arrow, honest copy about birthplace vs nationality.
- The compare support table has a caption and real `th` headers.

## Recommended order of work

Each step is its own small PR with tests, merged only after user sign-off:

1. **Data correctness** — dash-for-missing in Explorer and Team rows, correct lens links, formatted PA.
   *Done:* rows show "—"; the lens signature and Compare board no longer draw a 0% marker for a missing
   value; covered by `web/tests/missing-values.test.mjs`.
2. **Touch targets and contrast** — 44px dock links and remove button, raise failing colours to tokens that
   pass AA, 12px text floor.
3. **Search combobox** — full keyboard model, close on Escape/blur, clean listbox contents.
4. **Semantics** — `<ol>` leaderboards, labelled row values, `aria-current`, hidden glyphs.
5. **CSS consolidation** — remove dead families, collapse repeated owners, move to the five-range breakpoint
   set, replace literals with tokens. Done last and in slices, because it touches every page.
6. **Loading/error screens and Compare URL state.**

## Regression checks to add

Node tests in `web/tests/` (same approach as Media Vault's `libraryListRefinement.test.js`), asserting:

- No component renders a percentile with `?? 0` for display.
- Dock links, `.compare-remove` and other interactive controls declare at least 44px height/width.
- No `font-size` below the agreed floor outside an allow-list.
- Only the agreed breakpoint values appear in `globals.css`.
- No class in `globals.css` is unreferenced by the app.

CI (`.github/workflows/ci.yml`) and Vercel checks verify only what they run; they do not replace rendered
browser testing.

## Manual sign-off before merging any redesign PR

- Check Home, Players (each lens and search), a player profile, Compare (2 and 4 players), Blue Jays, Canada,
  Guide, 404 and an error state at **320, 375, 390, 430, 520, 768, 1024 and ≥1200px**.
- Confirm no horizontal page scroll and no clipped dock labels at 320px.
- Keyboard-only pass: skip link, nav, search (arrows/Escape/Enter), lens tabs, Compare add/remove/submit.
- Screen-reader spot check (VoiceOver or NVDA) on a Players row and the Compare board.
- 200% browser zoom and at least one real touch device.

**Policy:** no redesign PR is merged without explicit user approval.
