#!/usr/bin/env -S bun run
/**
 * Standalone HTTP server for MCP mock testing.
 * Imports createMcpApp from stackone-ai-node vendor submodule.
 *
 * Usage:
 *   ./tests/mocks/serve.ts [port]
 *   # or
 *   bun run tests/mocks/serve.ts [port]
 */
import { Hono } from "hono";
import { cors } from "hono/cors";
import {
  accountMcpTools,
  createMcpApp,
  defaultMcpTools,
  exampleBamboohrTools,
  mixedProviderTools,
} from "../../vendor/stackone-ai-node/mocks/mcp-server";

const port = parseInt(process.env.PORT || Bun.argv[2] || "8787", 10);

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
});

// Create the main app with CORS and mount the MCP app
const app = new Hono();

// Add CORS for cross-origin requests
app.use("/*", cors());

// Health check endpoint
app.get("/health", (c) => c.json({ status: "ok" }));

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

export default {
  port,
  fetch: app.fetch,
};

console.log(`MCP Mock Server running at http://localhost:${port}`);
console.log("Endpoints:");
console.log(`  - GET  /health       - Health check`);
console.log(`  - ALL  /mcp          - MCP protocol endpoint`);
console.log(`  - POST /actions/rpc  - RPC execution endpoint`);
