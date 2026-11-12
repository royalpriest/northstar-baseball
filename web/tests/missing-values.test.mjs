import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { barWidth, integer, isMissing, percentileValue } from "../lib/format.ts";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("compact percentiles show a dash for missing values, never zero", () => {
  for (const missing of [null, undefined, NaN]) {
    assert.equal(isMissing(missing), true);
    assert.equal(percentileValue(missing), "—");
    assert.equal(barWidth(missing), "0%");
  }
  assert.equal(isMissing(0), false);
  assert.equal(percentileValue(0), "0");
  assert.equal(percentileValue(49.6), "50");
  assert.equal(percentileValue(104), "100");
  assert.equal(barWidth(-3), "0%");
  assert.equal(barWidth(72.5), "72.5%");
  assert.equal(barWidth(140), "100%");
});

test("plate appearances use the shared integer format", () => {
  assert.equal(integer(1024), "1,024");
  assert.equal(integer(null), "—");
});

test("no row or chart coerces a missing percentile to zero for display", async () => {
  for (const path of [
    "components/player-explorer-row.tsx",
    "components/team-context-row.tsx",
    "components/lens-signature.tsx",
    "components/compare-workbench.tsx",
  ]) {
    const code = await source(path);
    assert.doesNotMatch(code, /Math\.round\([^)]*\?\?\s*0\)/, `${path} rounds a missing value to 0`);
    assert.doesNotMatch(code, /value \?\? 0\)\)\}%/, `${path} draws a missing value as a 0% bar`);
  }
});

test("bar fills and markers are only drawn when the percentile exists", async () => {
  const lens = await source("components/lens-signature.tsx");
  assert.match(lens, /available \? \(\s*<>\s*<span className="lens-line__fill"/);

  const compare = await source("components/compare-workbench.tsx");
  assert.match(compare, /available \? \(\s*<>\s*<span\s+className=\{`comparison-board__fill/);
  assert.match(compare, /\{integer\(player\.season_plate_appearances \?\? player\.plate_appearances\)\} PA/);
});

test("home lens cards open the matching Players lens", async () => {
  const home = await source("app/page.tsx");
  for (const lens of ["selection", "contact", "damage"]) {
    assert.match(
      home,
      new RegExp(`href="/players\\?lens=${lens}" className="lens-definition lens-definition--${lens}"`),
    );
  }
  assert.doesNotMatch(home, /href="\/players" className="lens-definition/);
});
