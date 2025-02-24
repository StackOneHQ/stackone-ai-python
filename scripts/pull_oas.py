# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "httpx",
#   "pyyaml",
#   "requests",
#   "beautifulsoup4"
# ]
# ///

import asyncio
import json
from pathlib import Path

import httpx
import requests
import yaml
from bs4 import BeautifulSoup

STACKONE_DOCS_BASE = "https://docs.stackone.com"
STACKONE_DOCS_URL = f"{STACKONE_DOCS_BASE}/openapi"
OAS_DIR = Path("stackone_ai/oas")


def get_api_specs() -> dict[str, str]:
    """Scrape OpenAPI spec URLs and their IDs from the documentation page"""
    response = requests.get(STACKONE_DOCS_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    specs = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/openapi/"):
            # Extract the ID and name from the link
            spec_id = href.split("/")[-1]
            # Parse the name from the link text (e.g., "CRM - v1.0" -> "crm")
            name = a.text.split("-")[0].strip().lower()
            if name == "stackone":
                name = "core"
            specs[name] = spec_id

    return specs


async def fetch_oas_spec(client: httpx.AsyncClient, spec_id: str) -> dict:
    """Fetch OpenAPI spec using its ID"""
    url = f"{STACKONE_DOCS_BASE}/openapi/{spec_id}"
    response = await client.get(url)
    response.raise_for_status()

    # Try both JSON and YAML parsing since specs can be in either format
    try:
        return response.json()
    except json.JSONDecodeError:
        return yaml.safe_load(response.text)


async def main() -> None:
    # Create output directory if it doesn't exist
    OAS_DIR.mkdir(parents=True, exist_ok=True)

    # add .gitignore
    (OAS_DIR / ".gitignore").write_text("*")

    # Get specs and their IDs from the documentation page
    specs = get_api_specs()
    print(f"Found {len(specs)} API specs to download:")
    for name, spec_id in specs.items():
        print(f"  - {name} ({spec_id})")

    async with httpx.AsyncClient() as client:
        for name, spec_id in specs.items():
            try:
                spec = await fetch_oas_spec(client, spec_id)

                # Save only as JSON since we're bundling with package
                json_path = OAS_DIR / f"{name}.json"
                json_path.write_text(json.dumps(spec, indent=2))

                print(f"✓ Downloaded {name} spec")

            except Exception as e:
                print(f"✗ Failed to download {name} spec: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
