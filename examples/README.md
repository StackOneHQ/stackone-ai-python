# StackOne AI SDK Examples

## Setup

Install dependencies:

```bash
uv sync --all-extras
```

Set your credentials using either approach:

**Option A: Environment variables (shell)**

```bash
export STACKONE_API_KEY=your-stackone-api-key
export STACKONE_ACCOUNT_ID=your-account-id
export OPENAI_API_KEY=your-openai-api-key  # for OpenAI/LangChain/CrewAI examples
```

**Option B: `.env` file**

```bash
cp .env.example .env
# Edit .env with your keys
```

## Running Examples

```bash
uv run examples/search_tools.py
```

Test all examples (no API keys needed):

```bash
uv run pytest examples
```

## Examples

| File | Description |
|------|-------------|
| `openai_integration.py` | OpenAI function calling with Workday tools |
| `langchain_integration.py` | LangChain tools integration |
| `crewai_integration.py` | CrewAI agent with Workday tools |
| `search_tools.py` | Tool discovery: direct fetch, semantic/local/auto search, search & execute |
| `auth_management.py` | API key and account ID configuration patterns |

## Environment Variables

See `.env.example` in the project root for all required variables.
