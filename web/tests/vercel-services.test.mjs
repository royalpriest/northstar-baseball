import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const config = JSON.parse(
  await readFile(new URL("../../vercel.json", import.meta.url), "utf8"),
);

test("Vercel services keep FastAPI internal behind the Next.js service", async () => {
  assert.equal(config.services.web.root, "web");
  assert.equal(config.services.web.framework, "nextjs");

  assert.deepEqual(config.services.web.bindings, [
    {
      type: "service",
      service: "api",
      format: "url",
      env: "NORTHSTAR_API_URL",
    },
  ]);

  assert.equal(config.services.api.root, ".");
  assert.equal(config.services.api.framework, "fastapi");
  assert.equal(config.services.api.entrypoint, "app:app");
  await access(new URL("../../app.py", import.meta.url));

  assert.deepEqual(config.rewrites, [
    {
      source: "/(.*)",
      destination: {
        service: "web",
      },
    },
  ]);

  const publiclyRoutedServices = config.rewrites
    .map((rewrite) => rewrite.destination?.service)
    .filter(Boolean);

  assert.deepEqual(publiclyRoutedServices, ["web"]);
});
