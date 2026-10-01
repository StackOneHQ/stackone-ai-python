#!/usr/bin/env -S pnpm exec tsx
/**
 * Standalone HTTP server for MCP mock testing.
 * Serves the MCP endpoint every tool is listed from and executed on, plus /accounts.
 *
 * Usage:
 *   ./tests/mocks/serve.ts [port]
 *   # or
 *   pnpm mock-server [port]
 */
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import {
  accountMcpTools,
  createMcpApp,
  defaultMcpTools,
  exampleBamboohrTools,
  fileTools,
  metaLookalikeTools,
  mixedProviderTools,
} from "./mcp-server";

const port = parseInt(process.env.PORT || process.argv[2] || "8787", 10);

// On by default, as it is for a project with feedback enabled. MOCK_SUBMIT_FEEDBACK=off
// serves a project without it, so the SDK's handling of its absence can be tested.
const submitFeedback = process.env.MOCK_SUBMIT_FEEDBACK !== "off";

interface RecordedRequest {
  path: "/mcp";
  /** The query string the call was made with, e.g. "" or "?tool-mode=search_execute". */
  search: string;
  accountId: string | null;
  method: string;
  name?: string;
  arguments?: unknown;
}

// What actually reached the wire: each JSON-RPC payload exactly as the SDK serialised it,
// so a test of the SDK's wire shape sees nulls and key order as sent.
const recorded: RecordedRequest[] = [];

// Create the MCP app with all test tool configurations
const mcpApp = createMcpApp({
  accountTools: {
    default: defaultMcpTools,
    acc1: accountMcpTools.acc1,
    acc2: accountMcpTools.acc2,
    acc3: accountMcpTools.acc3,
    "test-account": accountMcpTools["test-account"],
    mixed: mixedProviderTools,
    files: fileTools,
    lookalike: metaLookalikeTools,
    "your-bamboohr-account-id": exampleBamboohrTools,
    "your-stackone-account-id": exampleBamboohrTools,
  },
  submitFeedback,
});

// Create the main app with CORS and mount the MCP app
const app = new Hono();

// Add CORS for cross-origin requests
app.use("/*", cors());

// Health check endpoint
app.get("/health", (c) => c.json({ status: "ok" }));

app.get("/__requests", (c) => c.json(recorded));
app.delete("/__requests", (c) => {
  recorded.length = 0;
  return c.json({ cleared: true });
});

app.use("/mcp", async (c, next) => {
  if (c.req.method === "POST") {
    const accountId = c.req.header("x-account-id") ?? null;
    try {
      const payload = (await c.req.raw.clone().json()) as unknown;
      const messages = Array.isArray(payload) ? payload : [payload];
      for (const message of messages as { method?: string; params?: Record<string, unknown> }[]) {
        if (message?.method !== "tools/call" && message?.method !== "tools/list") continue;
        recorded.push({
          path: "/mcp",
          search: new URL(c.req.url).search,
          accountId,
          method: message.method,
          name: message.params?.name as string | undefined,
          arguments: message.params?.arguments,
        });
      }
    } catch {
      // Not JSON: the MCP handler rejects it, so there is nothing to record.
    }
  }
  await next();
});

// The SDK discovers accounts here when none is supplied. Returned as a bare list
// with an inactive entry, matching the shape and statuses the real API serves.
app.get("/accounts", (c) =>
  c.json([
    { id: "default", provider: "testprovider", status: "active" },
    { id: "dead", provider: "brokenprovider", status: "error" },
  ]),
);

// Mount the MCP app (handles /mcp endpoint)
app.route("/", mcpApp);

console.log(`MCP Mock Server starting on port ${port}...`);

serve({ fetch: app.fetch, port });

console.log(`MCP Mock Server running at http://localhost:${port}`);
console.log("Endpoints:");
console.log(`  - GET  /health       - Health check`);
console.log(`  - ALL  /mcp          - MCP protocol endpoint`);
console.log(`  - GET  /accounts     - Linked accounts`);
console.log(`  - GET  /__requests  - tools/call requests received`);
