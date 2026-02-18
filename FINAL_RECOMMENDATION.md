# Final Recommendation: Playwright Plus MCP Authentication Solution

## Problem Summary

**Issue:** Symlink approach to share Chrome default profile doesn't work because:
1. Chrome locks profile directories (prevents multiple instances)
2. Regular Chrome is already using the default profile
3. Playwright Plus can't launch new Chrome with same profile = creates new empty profile

## Working Solution: Standard Playwright MCP Extension Mode

**Status:** ✅ Already configured and working perfectly

**Why It Works:**
- Connects to existing Chrome via browser extension
- No profile locking (uses existing Chrome instance)
- All Cursor windows share same Chrome = shared authentication
- No conflicts with regular Chrome usage

**Current Configuration:**
```json
{
  "playwright": {
    "command": "npx",
    "args": ["-y", "@playwright/mcp@latest", "--extension"],
    "env": {
      "PLAYWRIGHT_MCP_EXTENSION_TOKEN": "uYucJ9hj3HNJMbccGc_KCUJ-qkyhNB4qg8C-Yx7BHag"
    }
  }
}
```

## Recommendation: Hybrid Approach

**Use Both MCP Servers:**

1. **Standard Playwright MCP (Extension Mode)** - For authenticated sites
   - Perplexity Academic (requires login)
   - Any site needing authentication
   - Uses existing Chrome with default profile
   - Shared authentication across all windows

2. **Playwright Plus MCP** - For isolated automation
   - Sites that don't need authentication
   - Tasks requiring project isolation
   - Separate profiles per project

**Configuration:**
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--extension"],
      "env": {
        "PLAYWRIGHT_MCP_EXTENSION_TOKEN": "uYucJ9hj3HNJMbccGc_KCUJ-qkyhNB4qg8C-Yx7BHag"
      }
    },
    "playwright-plus": {
      "command": "npx",
      "args": [
        "-y",
        "@ai-coding-labs/playwright-mcp-plus@latest",
        "--project-isolation"
      ]
    }
  }
}
```

**Usage:**
- Use `mcp_playwright_*` tools for Perplexity (authenticated)
- Use `mcp_playwright-plus_*` tools for other automation (isolated)

## Alternative: Remove Symlink, Use Extension Mode Only

If you only need authenticated sites and don't need Playwright Plus features:

**Simplified Configuration:**
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--extension"],
      "env": {
        "PLAYWRIGHT_MCP_EXTENSION_TOKEN": "uYucJ9hj3HNJMbccGc_KCUJ-qkyhNB4qg8C-Yx7BHag"
      }
    }
  }
}
```

Then remove the symlink:
```bash
rm ~/Library/Caches/ms-playwright/mcp-chrome-profile-shared
```

## Next Steps

1. **Decide on approach:**
   - Hybrid (both servers) - Best flexibility
   - Extension mode only - Simplest

2. **Update configuration accordingly**

3. **Test authentication:**
   - Use `mcp_playwright_browser_navigate` for Perplexity
   - Verify shared authentication across windows

---

**Bottom Line:** Symlink approach is blocked by Chrome's profile locking mechanism. Extension mode is the proven working solution for shared authentication.
