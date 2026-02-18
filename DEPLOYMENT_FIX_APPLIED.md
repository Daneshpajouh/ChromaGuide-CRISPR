# Playwright Plus MCP Configuration Fix

## Issue Identified

**Problem:** Playwright Plus MCP with `--project-isolation` was creating separate profile subdirectories per project, even when `--user-data-dir` was set to the symlink.

**Root Cause:** `--project-isolation` flag overrides `--user-data-dir` behavior by creating isolated profiles per project path, which defeats the purpose of sharing authentication.

## Solution Applied

**Removed `--project-isolation` flag** from Playwright Plus MCP configuration.

**Rationale:**
- Without `--project-isolation`, all Cursor windows will use the same `--user-data-dir`
- The symlink points to Chrome default profile (with authentication)
- All windows will share the same profile = shared authentication
- Browser contexts can still provide isolation at the context level (not profile level)

## Updated Configuration

**File:** `~/.cursor/mcp.json`

**Before:**
```json
"playwright-plus": {
  "args": [
    "-y",
    "@ai-coding-labs/playwright-mcp-plus@latest",
    "--project-isolation",
    "--user-data-dir=/Users/studio/Library/Caches/ms-playwright/mcp-chrome-profile-shared"
  ]
}
```

**After:**
```json
"playwright-plus": {
  "args": [
    "-y",
    "@ai-coding-labs/playwright-mcp-plus@latest",
    "--user-data-dir=/Users/studio/Library/Caches/ms-playwright/mcp-chrome-profile-shared"
  ]
}
```

## Next Steps

1. **Restart Cursor** again to load new configuration
2. **Test:** Navigate to Perplexity - should use default Chrome profile with authentication
3. **Verify:** All Cursor windows share authentication

## Trade-offs

**What We Gain:**
- ✅ Shared authentication across all windows
- ✅ Uses Chrome default profile (via symlink)
- ✅ Persistent logins

**What We Lose:**
- ❌ No automatic profile isolation per project
- ⚠️ All windows share same profile completely
- ⚠️ Still have context-level isolation (separate browser contexts per project)

**Note:** Browser contexts provide isolation for:
- Separate cookies/sessions per context (if needed)
- Separate localStorage per context
- But all contexts share the same base profile (authentication, saved passwords, etc.)

This is actually the desired behavior for shared authentication!

---

**Fix Applied:** December 14, 2025
**Status:** Configuration updated, awaiting Cursor restart
