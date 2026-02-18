# Symlink Solution Analysis - Why It's Not Working

## Current Situation

**Two Chrome Instances Running:**
1. **Regular Chrome (PID 58372)**
   - Using: `~/Library/Application Support/Google/Chrome/Default` (default profile)
   - No `--user-data-dir` flag (uses default location)
   - Has Perplexity authentication

2. **Playwright Plus Chrome (PID 79704)**
   - Using: `/Users/studio/Library/Caches/ms-playwright/mcp-chrome-profile-shared` (symlink)
   - Symlink points to: `~/Library/Application Support/Google/Chrome/Default`
   - Same physical directory as regular Chrome

**The Conflict:**
- Both Chrome instances try to use the same profile directory
- Chrome locks profile directory to prevent corruption
- When second instance tries to access locked profile, Chrome either:
  - Blocks it (causes "already in use" error)
  - OR creates a new empty profile subdirectory

**Why New Profile Was Created:**
When Chrome detects profile is locked, it may create a new profile or use a temporary one, which explains why you saw a new empty profile instead of the default one with authentication.

## Root Cause

**Chrome Profile Locking:** Chrome prevents multiple instances from using the same profile directory simultaneously. This is by design to prevent:
- Data corruption
- Concurrent database access conflicts
- Cookie/storage race conditions

## Solutions

### Option 1: Use Standard Playwright MCP Extension Mode (RECOMMENDED)

**Already Working:**
- ✅ Connects to existing Chrome (no profile locking)
- ✅ Uses default Chrome profile (via extension bridge)
- ✅ Shared authentication across all Cursor windows
- ✅ No conflicts with regular Chrome usage

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
    }
  }
}
```

**Trade-off:** Loses Playwright Plus features (project isolation, etc.)

### Option 2: Connect Playwright Plus to Existing Chrome via CDP

**Approach:** Use `--cdp-endpoint` flag to connect to existing Chrome instead of launching new one.

**Requirements:**
- Regular Chrome must be launched with `--remote-debugging-port`
- Playwright Plus connects via CDP
- No profile locking (only one Chrome instance)

**Issue:** Regular Chrome wasn't launched with debugging port enabled.

### Option 3: Copy Authentication Periodically

**Approach:**
1. Create separate Playwright profile
2. Periodically copy cookies/authentication from Chrome default profile
3. More complex, requires automation

### Option 4: Use Separate Profile for Automation

**Approach:**
1. Copy Chrome default profile to new location
2. Use that for Playwright Plus
3. Sync authentication periodically
4. More maintenance overhead

## Recommendation

**Use Standard Playwright MCP with Extension Mode:**
- ✅ Already configured and working
- ✅ No profile conflicts
- ✅ Shared authentication works
- ✅ Simple and reliable
- ❌ No Playwright Plus features

**OR**

**Keep Both:** Use standard Playwright MCP for authenticated sites (Perplexity) and Playwright Plus for other automation tasks that don't require authentication.

---

**Conclusion:** The symlink approach doesn't work because Chrome's profile locking prevents multiple instances from using the same profile directory. Extension mode avoids this by connecting to existing Chrome instead of launching new instances.
