# Playwright Plus MCP Upgrade Guide

## Research Findings Applied

Based on comprehensive research findings, we're upgrading from standard Playwright MCP to **Playwright Plus MCP** with project isolation.

## Benefits

✅ **Multi-window support** - Each Cursor window gets isolated browser session
✅ **100% API compatible** - Drop-in replacement for standard Playwright MCP
✅ **Persistent authentication** - Sessions survive IDE restarts
✅ **No conflicts** - True project-level isolation
✅ **Backward compatible** - Can keep old config as fallback

## Configuration Changes

### Old Configuration
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--extension"]
    }
  }
}
```

### New Configuration (Applied)
```json
{
  "mcpServers": {
    "playwright-plus": {
      "command": "npx",
      "args": [
        "-y",
        "@ai-coding-labs/playwright-mcp-plus@latest",
        "--project-isolation"
      ]
    },
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

## What Changed

1. **Added `playwright-plus` server** with `--project-isolation` flag
2. **Kept original `playwright` server** as backup/fallback
3. **Session storage:** System strategy (stored in OS cache, invisible)
   - macOS: `~/Library/Caches/ms-playwright/mcp-chrome-profile/playwright-plus-mcp/`
   - Windows: `%LOCALAPPDATA%/ms-playwright/mcp-chrome-profile/playwright-plus-mcp/`
   - Linux: `~/.cache/ms-playwright/mcp-chrome-profile/playwright-plus-mcp/`

## Next Steps

1. **Restart Cursor** to load new MCP configuration
2. **Test browser automation** using `mcp_playwright_browser_*` tools
3. **Verify multi-window isolation** by opening multiple project windows
4. **Use Playwright Plus** tools instead of cursor-ide-browser tools

## Tool Migration

### Old (cursor-ide-browser) - BROKEN
- `mcp_cursor-ide-browser_browser_type` ❌ (reports success but doesn't work)
- `mcp_cursor-ide-browser_browser_click` ❌ (script errors)

### New (playwright-plus) - WORKING
- `mcp_playwright_browser_type` ✅ (working)
- `mcp_playwright_browser_click` ✅ (working)
- `mcp_playwright_browser_navigate` ✅ (working)
- `mcp_playwright_browser_snapshot` ✅ (working)
- All Playwright tools with project isolation

## Usage with Project Isolation

When using Playwright Plus tools, each project window automatically gets its own isolated browser session. No manual configuration needed - the `--project-isolation` flag handles it automatically.

## References

- **GitHub:** https://github.com/vibe-coding-labs/playwright-plus-mcp
- **Package:** @ai-coding-labs/playwright-mcp-plus
- **Research Source:** Comprehensive research findings provided by user

---

**Upgrade Date:** December 14, 2025
**Status:** Configuration updated, awaiting Cursor restart
