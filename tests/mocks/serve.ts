#!/usr/bin/env -S pnpm exec tsx
/**
 * Standalone HTTP server for MCP mock testing.
 * Imports createMcpApp from stackone-ai-node vendor submodule.
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
  mixedProviderTools,
} from "./mcp-server";

const port = parseInt(process.env.PORT || process.argv[2] || "8787", 10);

// On by default, as it is for a project with feedback enabled. MOCK_SUBMIT_FEEDBACK=off
// serves a project without it, so the SDK's handling of its absence can be tested.
const submitFeedback = process.env.MOCK_SUBMIT_FEEDBACK !== "off";

interface RecordedRequest {
  path: "/mcp" | "/actions/rpc";
  accountId: string | null;
  method?: string;
  name?: string;
  arguments?: unknown;
  action?: string;
}

// What actually reached the wire. A handler only sees arguments after zod has parsed
// them, which strips unknown keys and hides whether a key was sent as null — the very
// things a test of the SDK's wire shape needs to see.
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
        if (message?.method !== "tools/call") continue;
        recorded.push({
          path: "/mcp",
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

// Every request, recorded before auth, so a refused one still counts as having been sent.
app.use("/actions/rpc", async (c, next) => {
  let action: string | undefined;
  try {
    action = ((await c.req.raw.clone().json()) as { action?: string }).action;
  } catch {
    action = undefined;
  }
  recorded.push({ path: "/actions/rpc", accountId: c.req.header("x-account-id") ?? null, action });
  await next();
});

// Mount the MCP app (handles /mcp endpoint)
app.route("/", mcpApp);

// RPC endpoint for tool execution
app.post("/actions/rpc", async (c) => {
  const authHeader = c.req.header("Authorization");
  const accountIdHeader = c.req.header("x-account-id");

  // Check for authentication
  if (!authHeader || !authHeader.startsWith("Basic ")) {
    return c.json(
      { error: "Unauthorized", message: "Missing or invalid authorization header" },
      401,
    );
  }

  // Execution is account-scoped too. This endpoint used to accept anything with a
  // "Basic " prefix and no account at all, so the sibling of the bug that shipped —
  // an unscoped execution request — could not be caught by any test.
  if (!accountIdHeader) {
    return c.json(
      { error: "Bad Request", message: "Missing x-account-id header in request" },
      400,
    );
  }

  const body = (await c.req.json()) as {
    action?: string;
    body?: Record<string, unknown>;
    headers?: Record<string, string>;
    path?: Record<string, string>;
    query?: Record<string, string>;
  };

  // Validate action is provided
  if (!body.action) {
    return c.json({ error: "Bad Request", message: "Action is required" }, 400);
  }

  // Test action to verify x-account-id is sent as HTTP header
  if (body.action === "test_account_id_header") {
    return c.json({
      data: {
        httpHeader: accountIdHeader,
        bodyHeader: body.headers?.["x-account-id"],
      },
    });
  }

  // Return mock response based on action
  if (body.action === "bamboohr_get_employee") {
    return c.json({
      data: {
        id: body.path?.id || "test-id",
        name: "Test Employee",
        ...body.body,
      },
    });
  }

  if (body.action === "bamboohr_list_employees") {
    return c.json({
      data: [
        { id: "1", name: "Employee 1" },
        { id: "2", name: "Employee 2" },
      ],
    });
  }

  if (body.action === "test_error_action") {
    return c.json({ error: "Internal Server Error", message: "Test error response" }, 500);
  }

  // Default response for other actions
  return c.json({
    data: {
      action: body.action,
      received: {
        body: body.body,
        headers: body.headers,
        path: body.path,
        query: body.query,
      },
    },
  });
});

console.log(`MCP Mock Server starting on port ${port}...`);

serve({ fetch: app.fetch, port });

console.log(`MCP Mock Server running at http://localhost:${port}`);
console.log("Endpoints:");
console.log(`  - GET  /health       - Health check`);
console.log(`  - ALL  /mcp          - MCP protocol endpoint`);
console.log(`  - POST /actions/rpc  - RPC execution endpoint`);
