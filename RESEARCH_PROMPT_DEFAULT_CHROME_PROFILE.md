# Research Prompt: Using Default Chrome Profile with Multiple Playwright Plus MCP Sessions

## EXECUTIVE SUMMARY

**Problem:** Playwright Plus MCP with `--project-isolation` creates separate browser profiles per project, which means authentication state (cookies, localStorage, login sessions) is NOT shared. This breaks authentication-dependent workflows like Perplexity Academic where users need to be logged in.

**Goal:** Find a way to use the **default Chrome profile** (with saved logins/authentication) across multiple Playwright Plus MCP sessions while maintaining project isolation for non-authentication data.

**Constraint:** Must work with multiple Cursor windows/projects simultaneously without authentication conflicts.

---

## CONTEXT & BACKGROUND

### Current Setup

**Configuration:**
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
    }
  }
}
```

**How It Works:**
- Each project gets a unique browser profile based on project path hash
- Profiles stored in: `~/Library/Caches/ms-playwright/mcp-chrome-profile/playwright-plus-mcp/{hash}/`
- **Problem:** Each profile is empty - no saved authentication, cookies, or login state
- **Impact:** Perplexity Academic requires authentication, but each new session is logged out

### Authentication Requirements

**Perplexity Academic:**
- Requires user authentication (email/password or OAuth)
- Stores session in cookies and localStorage
- "Research" mode requires Pro account (also stored in profile)
- Console shows: `[ERROR] Not signed in with the identity provider`
- API calls return: `403 Forbidden` (authentication required)

**User's Default Chrome Profile:**
- Already has Perplexity logged in
- Cookies and authentication state preserved
- Works perfectly when using Chrome manually
- Needs to be accessible to Playwright Plus MCP

---

## RESEARCH OBJECTIVES

### Primary Goal

**Find a configuration or method that allows:**
1. ✅ Using default Chrome profile (or shared authentication profile)
2. ✅ Multiple simultaneous Playwright Plus MCP sessions
3. ✅ Shared authentication state across all sessions
4. ✅ No authentication conflicts between projects
5. ✅ Persistent logins survive browser/IDE restarts

### Secondary Goals

1. Understand if Playwright Plus MCP supports custom profile paths
2. Determine if standard Playwright MCP (with extension) is better for this use case
3. Find ways to copy/import authentication from default profile
4. Investigate cookie/session injection methods
5. Explore hybrid approaches (shared auth + isolated storage)

---

## SPECIFIC RESEARCH QUESTIONS

### Question 1: Does Playwright Plus MCP Support Custom Profile Paths?

**What We Need to Know:**
- Can we specify `--user-data-dir` flag to point to default Chrome profile?
- Does `--project-isolation` conflict with custom profile paths?
- What happens if multiple projects use the same profile?
- Are there flags like `--profile-directory` or `--use-default-profile`?

**Evidence to Find:**
- Playwright Plus MCP CLI flags documentation
- Configuration options for profile management
- Examples of custom profile usage
- Known limitations or conflicts

### Question 2: Can We Use Standard Playwright MCP with Extension + Persistent Profile?

**What We Need to Know:**
- Standard Playwright MCP has `--extension` flag that connects to existing Chrome
- Does this work with default Chrome profile?
- Can multiple MCP servers connect to the same Chrome instance?
- Does this solve the authentication problem while maintaining multi-window support?

**Current Standard Config (from user's setup):**
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
        "PLAYWRIGHT_MCP_EXTENSION_TOKEN": "..."
      }
    }
  }
}
```

**Questions:**
- Does `--extension` mode use the user's default Chrome profile?
- Can we run this alongside Playwright Plus for different use cases?
- What are the limitations of extension mode?

### Question 3: Profile Sharing Strategies

**What We Need to Know:**
- Can we configure Playwright Plus to use a shared profile directory for authentication only?
- Is there a way to symlink/copy authentication data from default profile?
- Can we use Chrome's profile isolation features to share auth but isolate other data?
- Are there Playwright APIs to copy cookies/sessions between profiles?

**Technical Approaches to Investigate:**
1. **Shared Auth Profile:**
   - Create single "auth" profile with logins
   - Multiple projects reference this profile
   - Use Playwright's context isolation for project-specific data

2. **Cookie Injection:**
   - Export cookies from default Chrome profile
   - Inject into each new Playwright Plus session
   - Automate cookie sync on session start

3. **Profile Cloning:**
   - Clone default Chrome profile on first use
   - Each project gets a copy with authentication
   - Periodically sync auth updates

4. **Context-Level Sharing:**
   - Use single browser instance (shared auth)
   - Create isolated contexts per project
   - Contexts share browser cookies but isolate storage

### Question 4: Playwright Browser Contexts vs Profiles

**What We Need to Know:**
- Playwright supports browser contexts (isolated within same browser instance)
- Do contexts share authentication cookies from the browser?
- Can we use one browser instance (with default profile) and multiple contexts?
- Does Playwright Plus MCP support context-level isolation instead of profile-level?

**Key Concepts:**
- Browser instance = one Chrome process (shares profile/auth)
- Browser context = isolated session within browser (can share cookies)
- Profile = persistent storage directory (cookies, localStorage, etc.)

**Question:** Can Playwright Plus MCP be configured to use contexts instead of profiles?

### Question 5: Hybrid Configuration Strategies

**What We Need to Know:**
- Can we run BOTH standard Playwright MCP (extension mode, default profile) AND Playwright Plus MCP?
- Use standard for authenticated sites (Perplexity)
- Use Playwright Plus for isolated automation (other sites)
- Both available simultaneously via different MCP server names

**Configuration Example:**
```json
{
  "mcpServers": {
    "playwright-auth": {
      "command": "npx",
      "args": ["@playwright/mcp@latest", "--extension"],
      // Uses default Chrome profile with authentication
    },
    "playwright-plus": {
      "command": "npx",
      "args": ["@ai-coding-labs/playwright-mcp-plus@latest", "--project-isolation"],
      // Isolated profiles for other automation
    }
  }
}
```

**Questions:**
- Does this work? Any conflicts?
- How to choose which tool to use for which task?
- Performance implications of two MCP servers?

---

## TECHNICAL DETAILS TO RESEARCH

### Playwright Plus MCP Architecture

**What We Know:**
- Uses project path to generate unique profile hash
- Stores profiles in OS cache directory
- Each project = separate browser instance = separate profile

**What We Need:**
1. Source code analysis of profile creation logic
2. CLI flag reference (all available options)
3. Configuration file support (if any)
4. Environment variables for customization
5. API/extension points for profile management

### Chrome Profile Structure

**Default Chrome Profile Location:**
- macOS: `~/Library/Application Support/Google/Chrome/Default`
- Windows: `%LOCALAPPDATA%\Google\Chrome\User Data\Default`
- Linux: `~/.config/google-chrome/Default`

**Key Files/Folders:**
- `Cookies` - Authentication cookies
- `Local Storage/` - localStorage data (includes auth tokens)
- `Session Storage/` - Session-specific data
- `Preferences` - Browser settings, saved logins
- `Login Data` - Saved passwords (encrypted)
- `Web Data` - Autofill and other data

**Research Questions:**
- Which files are needed for Perplexity authentication?
- Can we safely copy specific files/directories?
- Are there locking issues when multiple processes access same profile?
- Chrome allows `--remote-debugging-port` - can Playwright use this?

### Playwright Profile Configuration

**Current Understanding:**
- Playwright can launch with custom `userDataDir`
- Can specify `profile` subdirectory within userDataDir
- Browser contexts can be created with specific storage state (cookies/localStorage)

**Research Needed:**
1. How to configure Playwright Plus to use specific userDataDir
2. Can we point it to Chrome's default profile directory?
3. What about Chrome profile locking (multiple instances)?
4. Does Playwright support "connect to existing browser" mode?

---

## ALTERNATIVE APPROACHES TO INVESTIGATE

### Approach 1: Playwright Plus with Custom Profile Flag

**Hypothesis:** Playwright Plus might support a flag to specify custom profile directory.

**Research:**
- Check if `--user-data-dir` or `--profile-path` flags exist
- Test if we can override the auto-generated profile hash
- Determine if custom profiles work with project isolation

### Approach 2: Standard Playwright MCP Extension Mode

**Hypothesis:** Extension mode connects to existing Chrome, which uses default profile.

**Research:**
- Verify extension mode uses default Chrome profile
- Test multiple MCP connections to same Chrome instance
- Determine if extension mode has multi-window/project support
- Check if extension token can be shared or must be unique

### Approach 3: Cookie/Session Export/Import

**Hypothesis:** We can export authentication from default profile and inject into Playwright profiles.

**Research:**
- Tools/methods to export Chrome cookies
- Playwright APIs for importing cookies (context.storageState)
- Automation script to sync cookies on session start
- Browser extension for cookie management

### Approach 4: Shared Auth Profile with Context Isolation

**Hypothesis:** Use one browser instance (with default profile) and create isolated contexts.

**Research:**
- Playwright Plus support for context-based isolation
- How to configure shared browser with multiple contexts
- Context isolation capabilities (storage, cookies, etc.)
- Performance comparison vs profile isolation

### Approach 5: Profile Symlinking/Copying

**Hypothesis:** Create symlinks or periodic copies of auth data from default profile.

**Research:**
- Chrome profile structure for authentication data
- Symlink feasibility (Chrome locking behavior)
- Scripts to copy auth files on startup
- Automation for keeping profiles in sync

---

## EXPECTED DELIVERABLES

### Required Findings

1. **Configuration Solution (If Exists)**
   - Exact CLI flags or config needed
   - Step-by-step setup instructions
   - Verification steps
   - Known limitations

2. **Alternative Solutions (If No Direct Config)**
   - Workarounds with detailed steps
   - Automation scripts if needed
   - Trade-offs and considerations

3. **Comparison Matrix**
   - Standard Playwright MCP (extension) vs Playwright Plus MCP
   - Profile isolation vs context isolation
   - Shared auth approaches ranked by feasibility

4. **Best Practice Recommendation**
   - Recommended approach for this use case
   - Why it's the best solution
   - Implementation guide

### Documentation Needed

1. **Playwright Plus MCP:**
   - Complete CLI flag reference
   - Configuration options
   - Profile management APIs
   - Extension/customization points

2. **Standard Playwright MCP:**
   - Extension mode documentation
   - Profile behavior in extension mode
   - Multi-session capabilities
   - Limitations

3. **Playwright Browser Contexts:**
   - Context vs profile isolation
   - Cookie/storage sharing
   - Context creation APIs
   - Best practices

4. **Chrome Profile Management:**
   - Profile structure for authentication
   - Safe copying/syncing methods
   - Locking behavior
   - Multi-instance considerations

---

## CONSTRAINTS & REQUIREMENTS

### Must Have

- ✅ Use default Chrome profile (or equivalent authentication state)
- ✅ Work with multiple Cursor windows simultaneously
- ✅ Persistent authentication (survives restarts)
- ✅ No manual authentication steps per session
- ✅ Compatible with Playwright MCP tools

### Nice to Have

- ✅ Project-level isolation for non-auth data
- ✅ No profile conflicts
- ✅ Good performance
- ✅ Simple configuration

### Cannot Do

- ❌ Manual login per session
- ❌ Breaking existing Chrome profile
- ❌ Requiring Chrome to be pre-launched
- ❌ Complex setup that breaks frequently

---

## RESEARCH SOURCES TO CHECK

### Official Documentation

1. **Playwright Plus MCP:**
   - GitHub: https://github.com/vibe-coding-labs/playwright-plus-mcp
   - README and documentation
   - Issue tracker (search for "profile", "authentication", "default")
   - CLI help/usage: `npx @ai-coding-labs/playwright-mcp-plus --help`

2. **Standard Playwright MCP:**
   - GitHub: https://github.com/microsoft/playwright-mcp
   - Documentation on extension mode
   - Profile and authentication handling
   - Multi-session support

3. **Playwright Documentation:**
   - Browser contexts: https://playwright.dev/docs/browser-contexts
   - Profiles and user data: https://playwright.dev/docs/browsers
   - Storage state: https://playwright.dev/docs/state
   - Connecting to existing browser: https://playwright.dev/docs/browsers#connect-to-a-browser

### Community Resources

4. **GitHub Issues & Discussions:**
   - Search Playwright Plus MCP issues for "profile", "auth", "default"
   - Search Playwright issues for "default chrome profile", "shared auth"
   - Community discussions on profile management

5. **Stack Overflow & Forums:**
   - Playwright questions about Chrome profiles
   - Browser automation authentication patterns
   - Multi-session browser management

6. **Blog Posts & Tutorials:**
   - Playwright authentication best practices
   - Browser profile management guides
   - Multi-instance browser automation

### Code & Implementation

7. **Source Code Analysis:**
   - Playwright Plus MCP source (profile creation logic)
   - Standard Playwright MCP source (extension mode)
   - Playwright core (browser context implementation)

8. **Example Configurations:**
   - Real-world MCP configurations
   - Playwright automation examples with profiles
   - Browser context isolation examples

---

## TESTING SCENARIOS TO VALIDATE SOLUTIONS

### Test 1: Authentication Persistence
- Configure solution
- Login to Perplexity in first session
- Open second Cursor window/project
- Verify second session is also logged in
- Restart Cursor
- Verify authentication persists

### Test 2: Multi-Session Isolation
- Open 3 Cursor windows with different projects
- All should be logged into Perplexity
- Perform different searches in each
- Verify no interference between sessions
- Verify each can work independently

### Test 3: Profile Integrity
- Use solution for extended period
- Verify Chrome default profile not corrupted
- Verify saved logins still work in regular Chrome
- Check for profile locking issues
- Monitor for performance degradation

### Test 4: Configuration Stability
- Solution should work after Cursor updates
- Should survive system restarts
- Should handle MCP server restarts gracefully
- Should not require frequent reconfiguration

---

## SUCCESS CRITERIA

### Solution is Successful If:

1. ✅ Default Chrome profile authentication accessible to Playwright Plus MCP
2. ✅ Multiple Cursor windows can use same authentication simultaneously
3. ✅ No manual login required after initial setup
4. ✅ Authentication persists across IDE/browser restarts
5. ✅ No conflicts between multiple sessions
6. ✅ Solution is stable and maintainable
7. ✅ Documentation is clear and complete

### Solution Quality Metrics

- **Ease of Setup:** ≤ 15 minutes from start to working
- **Reliability:** Works 99%+ of the time
- **Performance:** No noticeable slowdown vs isolated profiles
- **Maintenance:** Minimal ongoing intervention required
- **Compatibility:** Works with latest Playwright Plus MCP versions

---

## PRIORITY RESEARCH AREAS

### HIGHEST PRIORITY

1. **Playwright Plus MCP Profile Configuration**
   - Check if custom profile paths are supported
   - Look for flags like `--user-data-dir`, `--profile-path`
   - Verify if project isolation can work with shared profiles

2. **Standard Playwright MCP Extension Mode**
   - Verify it uses default Chrome profile
   - Test multi-session support
   - Compare with Playwright Plus for this use case

### HIGH PRIORITY

3. **Browser Context Isolation**
   - Can contexts share cookies but isolate storage?
   - Playwright Plus support for context-based isolation
   - Performance vs profile isolation

4. **Cookie/Session Management**
   - Export/import methods
   - Automation scripts
   - Sync strategies

### MEDIUM PRIORITY

5. **Profile Copying/Syncing**
   - Safe file copying methods
   - Automation approaches
   - Locking considerations

6. **Hybrid Configurations**
   - Running both MCP servers simultaneously
   - Use case routing strategies

---

## RESEARCH METHODOLOGY

### Phase 1: Documentation Review
- Read all official docs thoroughly
- Check GitHub READMEs and wikis
- Review changelogs for relevant features
- Look for examples and tutorials

### Phase 2: Source Code Analysis
- Examine Playwright Plus MCP source code
- Find profile creation/management logic
- Check for configuration options
- Identify extension points

### Phase 3: Community Research
- Search GitHub issues and discussions
- Check Stack Overflow questions
- Review blog posts and tutorials
- Look for similar use cases

### Phase 4: Testing & Validation
- Test promising solutions
- Document findings
- Identify limitations
- Provide workarounds if needed

### Phase 5: Solution Synthesis
- Compare all approaches
- Rank by feasibility and quality
- Provide recommended solution
- Create implementation guide

---

## EXPECTED TIMELINE

- **Phase 1-2:** 2-4 hours (documentation + source code)
- **Phase 3:** 2-3 hours (community research)
- **Phase 4:** 2-4 hours (testing promising solutions)
- **Phase 5:** 1-2 hours (documentation and synthesis)

**Total:** 7-13 hours of focused research

---

## OUTPUT FORMAT

### Research Report Should Include:

1. **Executive Summary**
   - Problem statement
   - Solution overview
   - Recommendation

2. **Detailed Findings**
   - All approaches investigated
   - Pros/cons of each
   - Technical details

3. **Recommended Solution**
   - Why this solution
   - Step-by-step implementation
   - Configuration files/code
   - Verification steps

4. **Alternative Solutions**
   - Other viable approaches
   - When to use them
   - Implementation guides

5. **Troubleshooting Guide**
   - Common issues
   - Solutions
   - Debugging tips

6. **References & Sources**
   - All documentation links
   - GitHub issues/repos
   - Community resources
   - Code examples

---

## CRITICAL QUESTIONS TO ANSWER

1. **Can Playwright Plus MCP use default Chrome profile?**
   - Yes/No/Partial answer
   - How to configure it
   - Any limitations

2. **Is Standard Playwright MCP Extension Mode Better for This?**
   - Comparison analysis
   - When to use which
   - Compatibility considerations

3. **What's the Simplest Working Solution?**
   - Identify easiest approach
   - Implementation complexity
   - Maintenance requirements

4. **Are There Any Gotchas?**
   - Profile locking issues
   - Security considerations
   - Performance impacts
   - Compatibility problems

5. **Future-Proof Solution?**
   - Will it work with updates?
   - Maintenance requirements
   - Deprecation risks

---

**Research Priority:** HIGH - Blocking automated Perplexity research workflow
**Urgency:** MEDIUM - Current solution works but requires manual authentication
**Complexity:** MEDIUM - Requires understanding Playwright internals and Chrome profiles
**Expected Outcome:** Working configuration that allows default Chrome profile with multiple Playwright Plus sessions

---

**END OF RESEARCH PROMPT**
