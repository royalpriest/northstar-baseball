import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

// Body of every @media (max-width: Npx) block, in source order.
function mediaBlocks(maxWidth) {
  const blocks = [];
  const opener = `@media (max-width: ${maxWidth}px) {`;
  let at = css.indexOf(opener);
  while (at !== -1) {
    let depth = 0;
    let i = at + opener.length - 1;
    for (; i < css.length; i += 1) {
      if (css[i] === "{") depth += 1;
      if (css[i] === "}") depth -= 1;
      if (depth === 0) break;
    }
    blocks.push({ start: at, body: css.slice(at + opener.length, i) });
    at = css.indexOf(opener, i);
  }
  return blocks;
}

test("home hero collapses to one column after its last two-column declaration", () => {
  const twoColumn = ".editorial-home .editorial-hero {\n  grid-template-columns: minmax(0, 1fr) minmax(300px, 0.72fr);";
  const lastTwoColumn = css.lastIndexOf(twoColumn);
  assert.ok(lastTwoColumn > 0, "expected a two-column hero declaration");

  const collapse = mediaBlocks(980).find(
    (block) =>
      block.start > lastTwoColumn &&
      /\.editorial-home \.editorial-hero \{\s*grid-template-columns: minmax\(0, 1fr\);/.test(block.body),
  );
  assert.ok(collapse, "a later <=980px rule must reset the hero to one column");
});

test("mobile dock is not trapped inside a backdrop-filtered header", () => {
  const dock = mediaBlocks(820).find((block) => block.body.includes(".site-nav__links {\n    position: fixed;"));
  assert.ok(dock, "expected the fixed mobile dock rule");
  assert.match(dock.body, /\.site-nav \{\s*backdrop-filter: none;/);
});

test("mobile dock links meet the 44px touch target", () => {
  for (const width of [820, 430]) {
    for (const block of mediaBlocks(width)) {
      const rule = block.body.match(/\.nav-link \{[^}]*\}/);
      if (!rule) continue;
      const height = rule[0].match(/min-height: (\d+)px/);
      if (height) assert.ok(Number(height[1]) >= 44, `nav-link at ${width}px is ${height[1]}px`);
    }
  }
});
