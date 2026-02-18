# MCP Configuration Simplified

## Changes Applied

**Removed:** Playwright Plus MCP
**Kept:** Standard Playwright MCP (Extension Mode)

## Reason

Playwright Plus MCP with symlink approach conflicted with Chrome's profile locking mechanism. Standard Playwright MCP with extension mode:
- ✅ Connects to existing Chrome (no conflicts)
- ✅ Uses default Chrome profile (shared authentication)
- ✅ Works across multiple Cursor windows
- ✅ Already proven and tested

## Current Configuration

**File:** `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": [
        "-y",
        "@playwright/mcp@latest",
        "--extension"
      ],
      "env": {
        "PLAYWRIGHT_MCP_EXTENSION_TOKEN": "uYucJ9hj3HNJMbccGc_KCUJ-qkyhNB4qg8C-Yx7BHag"
      }
    }
  }
}
```

## Usage

- Use `mcp_playwright_browser_*` tools for browser automation
- Perplexity Academic research will use shared Chrome authentication
- All Cursor windows share the same authentication state

## Next Steps

1. Continue deep research in Perplexity for latest CRISPRO-MAMBA-X architecture
2. Research latest Mamba-2, BiMamba, and CRISPR prediction methods
3. Verify implementation aligns with latest best practices

---

**Configuration Updated:** December 14, 2025
**Status:** Ready for Perplexity deep research
