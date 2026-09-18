/**
 * Mock MCP server for testing using Hono's app.request() method.
 * This creates an MCP-compatible handler that can be used with MSW
 * without starting a real HTTP server.
 */
import type { Hono as HonoApp } from 'hono';
import { StreamableHTTPTransport } from '@hono/mcp';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { Hono } from 'hono';
import { basicAuth } from 'hono/basic-auth';
import { z } from 'zod';

export interface McpToolDefinition {
	name: string;
	description?: string;
	inputSchema: {
		type: 'object';
		properties?: Record<string, unknown>;
		required?: string[];
		additionalProperties?: boolean;
	};
}

export interface MockMcpServerOptions {
	/** Tools available per account ID. Use 'default' for tools when no account header is provided. */
	accountTools: Record<string, readonly McpToolDefinition[]>;
}

/**
 * Creates an MSW handler for mocking MCP protocol requests.
 * Uses Hono's app.request() to handle requests without starting a server.
 *
 * @example
 * ```ts
 * import { server } from './mocks/node';
 * import { createMcpHandler, defaultMcpTools, accountMcpTools } from './mocks/mcp-server';
 *
 * // In your test setup
 * server.use(
 *   createMcpHandler({
 *     accountTools: {
 *       default: defaultMcpTools,
 *       'account-1': accountMcpTools.acc1,
 *     },
 *   })
 * );
 * ```
 */
export function createMcpApp(options: MockMcpServerOptions): HonoApp {
	const { accountTools } = options;

	// Create a Hono app that handles MCP protocol
	const app = new Hono();

	// Apply Basic Auth middleware with hardcoded test credentials
	app.use(
		'/mcp',
		basicAuth({
			username: 'test-key',
			password: '',
		}),
	);

	app.all('/mcp', async (c) => {
		// The real endpoint rejects a request with no account. Defaulting here made the
		// mock more permissive than production and hid a fatal SDK bug: `?? 'default'`
		// meant an SDK that never sent the header still got a full catalog.
		const accountId = c.req.header('x-account-id') ?? c.req.query('x-account-id');
		if (!accountId) {
			return c.json(
				{ statusCode: 400, message: 'Missing x-account-id header or query parameter in request' },
				400,
			);
		}
		// No `?? accountTools.default` fallback. Serving a full catalog for an account
		// that does not exist is the same permissiveness that hid the missing-header bug
		// one line further up: any bug that sends a wrong, stale or mangled account id
		// would be invisible. The real API refuses.
		if (!(accountId in accountTools)) {
			return c.json({ statusCode: 404, message: `Unknown account ${accountId}` }, 404);
		}
		let tools = accountTools[accountId] ?? [];

		// The real endpoint swaps the per-action catalog for two meta tools per
		// connector under this mode. Without it the SDK's search/execute path has
		// nothing to talk to and goes untested.

		// Create a new MCP server instance per request
		const mcp = new McpServer({ name: 'test-mcp-server', version: '1.0.0' });
		const searchExecute = c.req.query('tool-mode') === 'search_execute';
		const transport = new StreamableHTTPTransport();

		if (searchExecute) {
			mcp.registerTool(
				`mock_${accountId}_search_actions`,
				{
					description: 'Search for available actions in natural language.',
					inputSchema: { query: z.string(), top_k: z.number().optional() },
				},
				async () => ({
					content: [
						{
							type: 'text' as const,
							text: JSON.stringify({
								actions: [
									{
										action_id: 'mock_list_items',
										description: 'List items',
										example_request: { query: { page_size: 25 } },
									},
								],
							}),
						},
					],
				}),
			);
			mcp.registerTool(
				`mock_${accountId}_execute_action`,
				{
					description: 'Execute an action by its action_id.',
					inputSchema: {
						action_id: z.string(),
						path: z.record(z.string(), z.unknown()).optional(),
						query: z.record(z.string(), z.unknown()).optional(),
						body: z.record(z.string(), z.unknown()).optional(),
						headers: z.record(z.string(), z.string()).optional(),
					},
				},
				async ({ action_id, query }: { action_id: string; query?: Record<string, unknown> }) => {
					// An unknown action must come back as isError — a normal response with
					// the flag set — the way the real endpoint reports it.
					const known = action_id === 'mock_list_items';
					return {
						isError: !known,
						content: [
							{
								type: 'text' as const,
								text: JSON.stringify(
									known
										? { data: { nodes: [] }, echoed_query: query ?? null }
										: { error: `Unknown action ${action_id}` },
								),
							},
						],
					};
				},
			);
			await mcp.connect(transport);
			return transport.handleRequest(c);
		}

		for (const tool of tools) {
			mcp.registerTool(
				tool.name,
				{
					description: tool.description,
					// MCP SDK expects Zod-like schema but accepts JSON Schema objects
					// eslint-disable-next-line @typescript-eslint/no-explicit-any -- MCP SDK type mismatch
					inputSchema: tool.inputSchema as any,
				},
				async ({ params }: { params: { arguments?: Record<string, unknown> } }) => {
					const args = params.arguments ?? {};
					return { content: [], structuredContent: args, _meta: undefined };
				},
			);
		}

		await mcp.connect(transport);
		return transport.handleRequest(c);
	});

	return app;
}

// Pre-defined tool sets for common test scenarios

export const defaultMcpTools = [
	{
		name: 'default_tool_1',
		description: 'Default Tool 1',
		inputSchema: {
			type: 'object',
			properties: { fields: { type: 'string' } },
		},
	},
	{
		name: 'default_tool_2',
		description: 'Default Tool 2',
		inputSchema: {
			type: 'object',
			properties: { id: { type: 'string' } },
			required: ['id'],
		},
	},
] as const satisfies McpToolDefinition[];

export const accountMcpTools = {
	acc1: [
		{
			name: 'acc1_tool_1',
			description: 'Account 1 Tool 1',
			inputSchema: {
				type: 'object',
				properties: { fields: { type: 'string' } },
			},
		},
		{
			name: 'acc1_tool_2',
			description: 'Account 1 Tool 2',
			inputSchema: {
				type: 'object',
				properties: { id: { type: 'string' } },
				required: ['id'],
			},
		},
	],
	acc2: [
		{
			name: 'acc2_tool_1',
			description: 'Account 2 Tool 1',
			inputSchema: {
				type: 'object',
				properties: { fields: { type: 'string' } },
			},
		},
		{
			name: 'acc2_tool_2',
			description: 'Account 2 Tool 2',
			inputSchema: {
				type: 'object',
				properties: { id: { type: 'string' } },
				required: ['id'],
			},
		},
	],
	acc3: [
		{
			name: 'acc3_tool_1',
			description: 'Account 3 Tool 1',
			inputSchema: {
				type: 'object',
				properties: { fields: { type: 'string' } },
			},
		},
	],
	'test-account': [
		{
			name: 'dummy_action',
			description: 'Dummy tool',
			inputSchema: {
				type: 'object',
				properties: {
					foo: {
						type: 'string',
						description: 'A string parameter',
					},
				},
				required: ['foo'],
				additionalProperties: false,
			},
		},
	],
} as const satisfies Record<string, McpToolDefinition[]>;

/** Tools for the quickstart and example tests */
export const exampleBamboohrTools = [
	{
		name: 'bamboohr_list_employees',
		description: 'List all employees from BambooHR',
		inputSchema: {
			type: 'object',
			properties: {
				query: {
					type: 'object',
					properties: {
						limit: { type: 'number', description: 'Limit the number of results' },
					},
				},
			},
		},
	},
	{
		name: 'bamboohr_get_employee',
		description: 'Get a single employee by ID from BambooHR',
		inputSchema: {
			type: 'object',
			properties: {
				id: { type: 'string', description: 'The employee ID' },
				fields: { type: 'string', description: 'Fields to retrieve' },
			},
			required: ['id'],
		},
	},
	{
		name: 'bamboohr_create_employee',
		description: 'Create a new employee in BambooHR',
		inputSchema: {
			type: 'object',
			properties: {
				name: { type: 'string', description: 'Employee name' },
				personal_email: { type: 'string', description: 'Employee email' },
				department: { type: 'string', description: 'Department name' },
				start_date: { type: 'string', description: 'Start date' },
				hire_date: { type: 'string', description: 'Hire date' },
			},
			required: ['name'],
		},
	},
] as const satisfies McpToolDefinition[];

export const mixedProviderTools = [
	{
		name: 'hibob_list_employees',
		description: 'HiBob List Employees',
		inputSchema: {
			type: 'object',
			properties: { fields: { type: 'string' } },
		},
	},
	{
		name: 'hibob_create_employees',
		description: 'HiBob Create Employees',
		inputSchema: {
			type: 'object',
			properties: { name: { type: 'string' } },
			required: ['name'],
		},
	},
	{
		name: 'bamboohr_list_employees',
		description: 'BambooHR List Employees',
		inputSchema: {
			type: 'object',
			properties: { fields: { type: 'string' } },
		},
	},
	{
		name: 'bamboohr_get_employee',
		description: 'BambooHR Get Employee',
		inputSchema: {
			type: 'object',
			properties: { id: { type: 'string' } },
			required: ['id'],
		},
	},
	{
		name: 'workday_list_employees',
		description: 'Workday List Employees',
		inputSchema: {
			type: 'object',
			properties: { fields: { type: 'string' } },
		},
	},
] as const satisfies McpToolDefinition[];
