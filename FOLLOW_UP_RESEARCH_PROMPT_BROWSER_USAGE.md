# FOLLOW-UP RESEARCH PROMPT: Cursor IDE Built-in Browser Web Automation Solution
## Critical Investigation Required - Based on Failed Universal Solution Attempt

---

## EXECUTIVE SUMMARY

**Status:** Previous research prompt created, but additional testing has revealed the problem is MORE severe than initially thought. A universal JavaScript solution was attempted but cannot be implemented due to missing JavaScript execution capability. All native browser tools (`browser_type`, `browser_click`) are non-functional.

**Goal:** Find the CORRECT working method to use Cursor IDE built-in browser tools to interact with React SPA websites (specifically Perplexity Academic), OR determine if the tools are fundamentally broken and require fixes/enhancements.

**Urgency:** CRITICAL - This is blocking all automated research tasks for the project.

---

## FULL CONTEXT & BACKGROUND

### Project Requirements
- **AI Assistant:** Auto (operating in Cursor IDE)
- **Browser:** Cursor IDE built-in browser (MCP cursor-ide-browser) - **MANDATORY** (cannot use MCP Playwright browser)
- **Target Website:** Perplexity Academic (`https://www.perplexity.ai/academic`)
- **Required Tasks:**
  1. Navigate to Perplexity Academic
  2. Enable "Deep Research" mode (circuit board icon)
  3. Enable "Academic Sources" toggle
  4. Type and submit research queries
  5. Extract results

### Previous Research Prompt
An initial research prompt was created (`RESEARCH_PROMPT_BROWSER_USAGE.md`) based on initial failures. Since then, additional testing has been conducted with a "universal solution" that requires JavaScript execution - revealing a critical gap.

---

## AVAILABLE TOOLS (COMPLETE LIST)

### ✅ **Tools That WORK:**
1. `mcp_cursor-ide-browser_browser_navigate(url)` - **WORKS PERFECTLY**
   - Successfully navigates to any URL
   - Example: Navigated to `https://www.perplexity.ai/academic` ✅

2. `mcp_cursor-ide-browser_browser_snapshot()` - **WORKS PERFECTLY**
   - Captures full page accessibility tree
   - Returns structured YAML with roles, refs, children
   - File saved to `/Users/studio/.cursor/browser-logs/snapshot-{timestamp}.log`
   - Example: Captured 455 lines, 31056 bytes ✅

3. `mcp_cursor-ide-browser_browser_press_key(key)` - **WORKS PARTIALLY**
   - Navigation keys work (e.g., `"/"` focuses search box)
   - Example: `press_key("/")` successfully focuses search ✅
   - Status: Works for focus, but not for typing text

4. `mcp_cursor-ide-browser_browser_wait_for(time, text?, textGone?)` - **WORKS**
   - Waits specified time
   - Can wait for text appearance/disappearance
   - Example: `wait_for(time=3)` works ✅

5. `mcp_cursor-ide-browser_browser_console_messages()` - **WORKS**
   - Returns console logs
   - Example: Captured CORS errors, debug messages ✅

6. `mcp_cursor-ide-browser_browser_network_requests()` - **WORKS**
   - Returns all network activity
   - Example: Captured API calls, resource loads ✅

7. `mcp_cursor-ide-browser_browser_navigate_back()` - **NOT TESTED** (likely works)

8. `mcp_cursor-ide-browser_browser_resize(width, height)` - **NOT TESTED** (not needed)

9. `mcp_cursor-ide-browser_browser_hover(element, ref)` - **NOT TESTED**

10. `mcp_cursor-ide-browser_browser_select_option(element, ref, values)` - **NOT TESTED**

### ❌ **Tools That DON'T WORK:**

1. `mcp_cursor-ide-browser_browser_click(element, ref)` - **COMPLETELY BROKEN**
   - **Error:** "Script failed to execute, this normally means an error was thrown."
   - **Tested On:** Buttons, toggles, switches, icons, links
   - **Result:** Fails on EVERY attempt
   - **Example Attempts:**
     ```javascript
     // Attempt 1: Click Deep Research icon
     browser_click(element="Deep Research icon", ref="ref-cursor-el-182")
     // Result: Script execution error

     // Attempt 2: Click Academic Sources toggle
     browser_click(element="Academic Sources toggle", ref="ref-cursor-el-1281")
     // Result: Script execution error

     // Attempt 3: Click any button
     browser_click(element="Button name", ref="ref-from-snapshot")
     // Result: Script execution error (EVERY TIME)
     ```

2. `mcp_cursor-ide-browser_browser_type(element, ref, text, submit?)` - **FALSE POSITIVE FAILURE**
   - **Tool Reports:** Success (claims to type characters)
   - **Actual Result:** Nothing happens on page
   - **Evidence of Failure:**
     - Snapshot before/after: IDENTICAL (no changes)
     - No text appears in accessibility tree
     - No query text visible on page
     - No search API calls in network requests
     - URL unchanged
     - No results appear
   - **Tested Configurations:**
     - Slow typing mode (`slowly=true`)
     - Fast typing mode (default)
     - With `submit=true`
     - Without `submit`
     - Multiple element references
     - Different text inputs
   - **Example Attempt:**
     ```javascript
     // Step 1: Focus (WORKS)
     browser_press_key(key="/")  // ✅ Successfully focuses search

     // Step 2: Type (FAILS SILENTLY)
     browser_type(
       element="Focused search input",
       ref="ref-may16gi8er",
       text="Mamba-2 BiMamba genomics CRISPR 2024 2025",
       slowly=true,
       submit=true
     )
     // Tool reports: "Characters typed: 41, Form submitted: yes"
     // Reality: Page unchanged, no text visible, nothing happened
     ```

### ❌ **Missing Critical Tool:**

3. **NO JavaScript Execution Capability**
   - **Missing:** `browser_evaluate()`, `browser_run_code()`, or any JavaScript execution tool
   - **Impact:** Cannot inject JavaScript, cannot manipulate DOM directly, cannot execute scripts
   - **Blocking:** Universal automation solution that requires JavaScript injection

---

## DETAILED FAILURE ANALYSIS

### Failure 1: browser_type False Positives

**Test Configuration:**
- **Date:** December 14, 2025, 22:55 UTC
- **Page:** `https://www.perplexity.ai/academic`
- **Test:** Type query and submit
- **Steps:**
  1. Navigate to page ✅
  2. Wait 3 seconds ✅
  3. Take snapshot ✅
  4. Press "/" key to focus search ✅
  5. Wait 1 second ✅
  6. Type query with `browser_type()` ❌
  7. Wait 3 seconds ✅
  8. Take snapshot ✅

**Tool Response:**
```
### Action: type
- Characters typed: 41
- Typing mode: slow (character-by-character)
- Form submitted: yes (Enter key pressed)
```

**Actual Page State:**
- Snapshot before: `/Users/studio/.cursor/browser-logs/snapshot-2025-12-14T22-54-52-451Z.log`
  - File size: 31056 bytes
  - Lines: 455
- Snapshot after: `/Users/studio/.cursor/browser-logs/snapshot-2025-12-14T22-55-53-508Z.log`
  - File size: 31056 bytes
  - Lines: 455
- **Diff Result:** NO DIFFERENCES FOUND (files identical)
- **Network Requests:** No search query API calls (only normal page loads)
- **URL:** Unchanged (still `/academic`)
- **Accessibility Tree:** No query text visible

**Conclusion:** `browser_type()` reports success but makes ZERO changes to the page.

### Failure 2: browser_click Script Errors

**Test Configuration:**
- **Target Elements Tested:**
  - Deep Research mode button (circuit board icon)
  - Academic Sources toggle switch
  - Various buttons and links
  - Toggle switches in dropdown menus

**Error Message (CONSISTENT):**
```
Error executing tool browser_click: Script failed to execute, this normally means an error was thrown.
```

**Element Reference Format Used:**
- Tried exact ref from snapshot (e.g., `ref-cursor-el-182`)
- Tried ref without prefix (e.g., `cursor-el-182`)
- Tried element names from accessibility tree
- Tried different button types

**Result:** 100% failure rate - NO successful clicks achieved.

### Failure 3: Universal Solution Blocked

**Universal Solution Provided:**
A comprehensive JavaScript automation class (`UniversalInputAutomation`) was provided that:
- Finds input elements using multiple strategies
- Dispatches complete React event chains
- Handles contenteditable elements
- Works with custom React components

**Solution Requirements:**
```javascript
// Requires JavaScript execution
await UniversalInputAutomation.typeInAnyInput('query');
```

**Why It Can't Be Used:**
- No `browser_evaluate()` tool available
- Cannot inject JavaScript into page
- Cannot execute DOM manipulation code
- Cannot dispatch synthetic events

**Conclusion:** The universal solution is excellent, but requires capabilities the browser doesn't have.

---

## ELEMENT IDENTIFICATION CHALLENGES

### Perplexity Search Input Structure

**Accessibility Tree Shows:**
- Only `role: generic` elements (no `role: textbox`)
- Nested structure: `generic` → `generic` → `generic` → ... (deep nesting)
- No semantic input elements visible

**Actual DOM (from user feedback):**
```html
<p dir="auto"></p>
<!-- Inside nested divs with complex React component structure -->
```

**Search Attempts:**
- ✅ `role="textbox"` - NOT FOUND
- ✅ `role="searchbox"` - NOT FOUND
- ✅ `<input>` elements - NOT FOUND in snapshot
- ✅ `<textarea>` elements - NOT FOUND
- ✅ `contenteditable` attributes - NOT FOUND in snapshot
- ❓ Generic elements - Found, but which one is editable?

**Element Reference Examples from Snapshots:**
```yaml
- role: generic
  ref: ref-may16gi8er
  children:
    - role: generic
      ref: ref-u3d5md2tgb
      children:
        # ... more nesting
        # No clear indication which element accepts input
```

**Question:** How do I identify which `ref` corresponds to the editable input element?

---

## WHAT WE KNOW WORKS

### Verified Working Operations

1. **Navigation**
   ```javascript
   browser_navigate(url="https://www.perplexity.ai/academic")
   // ✅ Works perfectly
   ```

2. **Snapshot Capture**
   ```javascript
   browser_snapshot()
   // ✅ Captures full page state
   // Returns: Structured YAML with roles, refs, children
   ```

3. **Keyboard Focus**
   ```javascript
   browser_press_key(key="/")
   // ✅ Focuses search box (based on behavior - no visual confirmation possible)
   ```

4. **Waiting**
   ```javascript
   browser_wait_for(time=3)
   // ✅ Waits correctly
   ```

5. **Console & Network Access**
   ```javascript
   browser_console_messages()
   browser_network_requests()
   // ✅ Both return data successfully
   ```

### What This Tells Us

- The browser connection works
- Navigation works
- Page state can be captured
- Some keyboard input works (focus)
- But actual text input and clicking do NOT work

---

## SPECIFIC RESEARCH QUESTIONS

### Critical Questions (Must Answer)

1. **Why does `browser_type()` report success but not actually type?**
   - Is there a validation step missing?
   - Does the tool require specific element properties?
   - Is there a timing issue?
   - Does it work with standard HTML inputs but fail on React components?
   - Is this a known bug?

2. **How do I identify editable input elements in accessibility trees?**
   - When snapshot shows only `role: generic`, how do I find the input?
   - Should I look for specific parent/child relationships?
   - Are there hidden attributes in the snapshot?
   - How do contenteditable elements appear in accessibility trees?

3. **Why does `browser_click()` fail with script errors?**
   - What causes "Script failed to execute"?
   - Are there specific element types it cannot click?
   - Is there a different format for refs needed?
   - Is this a known limitation or bug?

4. **Is there an alternative way to interact with elements?**
   - Can keyboard navigation (Tab, Space, Enter) work for everything?
   - Are there keyboard shortcuts for Perplexity?
   - Can I use a different interaction pattern?

5. **What is the correct element reference format?**
   - Should I use `ref-may16gi8er` exactly as shown?
   - Or `may16gi8er` without prefix?
   - Do refs need to be prefixed differently?
   - Are refs stable or do they change?

6. **How do I verify if an action actually worked?**
   - Besides comparing snapshots, what verification methods exist?
   - Can I check network requests to confirm actions?
   - Are there console messages that indicate success/failure?

7. **Are these tools designed for React SPAs?**
   - Do the tools work with standard HTML but fail on React?
   - Are there known issues with React/Virtual DOM?
   - Is there a workaround for React components?

### Secondary Questions

8. **Can I use a pure keyboard approach?**
   - Type character-by-character using `press_key`?
   - Navigate with Tab, activate with Space/Enter?
   - Would this work or have the same issues?

9. **Are there browser tool updates or fixes available?**
   - Has this been reported as a bug?
   - Are there newer versions with fixes?
   - Are there workarounds documented?

10. **Is JavaScript execution capability planned or available elsewhere?**
    - Will `browser_evaluate` be added?
    - Is there a hidden/undocumented way to execute JS?
    - Are there alternative tools with JS execution?

---

## TECHNICAL SPECIFICATIONS

### Browser Details
- **Type:** Cursor IDE Built-in Browser
- **MCP Server:** `cursor-ide-browser`
- **Engine:** Chromium-based (inferred from behavior)
- **Accessibility API:** Uses ARIA/accessibility tree
- **Tool Version:** Latest (2024-2025)

### Page Details
- **URL:** `https://www.perplexity.ai/academic`
- **Framework:** React SPA
- **Input Type:** Custom React component (not standard HTML input)
- **DOM Structure:** Virtual DOM, dynamically generated
- **Element References:** Change per snapshot (dynamic)

### Snapshot Format
```yaml
- role: generic
  ref: ref-{randomstring}
  children:
    - role: button
      name: "Button Name"
      ref: ref-{differentrandomstring}
      children: [...]
```

### Error Patterns
- **Click Errors:** Always "Script failed to execute"
- **Type Errors:** Silent failure (reports success but doesn't work)
- **No Element Errors:** N/A (tools report success even when failing)

---

## TESTING EVIDENCE (COMPLETE)

### Test 1: browser_type Failure
**Command:**
```javascript
browser_press_key(key="/")
browser_wait_for(time=1)
browser_type(
  element="Focused search input",
  ref="ref-may16gi8er",
  text="Mamba-2 BiMamba genomics CRISPR 2024 2025",
  slowly=true,
  submit=true
)
browser_wait_for(time=3)
browser_snapshot()
```

**Tool Response:**
- Characters typed: 41
- Form submitted: yes
- Typing mode: slow

**Verification:**
- Snapshot before: 31056 bytes
- Snapshot after: 31056 bytes
- Diff: No differences
- Network: No search API calls
- Result: ❌ FAILED (false positive)

### Test 2: browser_click Failure
**Command:**
```javascript
browser_click(element="Deep Research icon", ref="ref-cursor-el-182")
```

**Tool Response:**
- Error: Script failed to execute

**Verification:**
- Snapshot: No changes
- Result: ❌ FAILED

### Test 3: Universal Solution Attempt
**Approach:** Inject JavaScript automation class

**Blocking Issue:**
- No `browser_evaluate()` tool available
- Cannot inject JavaScript
- Result: ❌ CANNOT ATTEMPT

---

## WHAT NEEDS TO BE FOUND

### Required Deliverables

1. **Working Solution for browser_type**
   - Step-by-step guide that actually works
   - Correct element identification method
   - Verification steps to confirm typing succeeded
   - Code examples that have been tested and verified

2. **Working Solution for browser_click**
   - Step-by-step guide that actually works
   - Correct ref format and usage
   - How to handle different element types
   - Code examples that have been tested and verified

3. **Element Identification Guide**
   - How to find editable inputs in accessibility trees
   - How to identify React components
   - How to work with contenteditable elements
   - Patterns for custom components

4. **Verification Methods**
   - How to confirm actions actually worked
   - What to check in snapshots
   - What network requests indicate success
   - Console message patterns

5. **Troubleshooting Guide**
   - Common issues and solutions
   - Known bugs and workarounds
   - Debugging techniques
   - Alternative approaches when tools fail

6. **If Tools Are Broken:**
   - Confirmation that tools are non-functional
   - Official bug reports or known issues
   - Timeline for fixes
   - Alternative tools or methods (within Cursor IDE built-in browser)

---

## CONSTRAINTS & REQUIREMENTS

### Absolute Requirements
- ✅ Must use **Cursor IDE built-in browser** (NOT MCP Playwright browser)
- ✅ Must work with **React SPA** websites
- ✅ Must handle **custom input components** (not just standard HTML)
- ✅ Must provide **verification methods**

### Nice to Have
- JavaScript execution capability
- Better error messages
- Debugging tools
- Element inspection capabilities

### Cannot Use
- ❌ MCP Playwright browser (user explicitly forbade)
- ❌ External automation tools
- ❌ Manual intervention (must be automated)

---

## RESEARCH APPROACH REQUESTED

Please research using multiple sources:

1. **Official Documentation**
   - Cursor IDE browser automation docs
   - MCP cursor-ide-browser specification
   - Release notes and changelogs
   - API reference documentation

2. **GitHub & Issue Trackers**
   - cursor-ide-browser repository issues
   - Cursor IDE issues related to browser automation
   - Bug reports matching these symptoms
   - Feature requests for missing capabilities

3. **Technical Forums & Communities**
   - Stack Overflow questions about Cursor IDE browser
   - Cursor IDE community discussions
   - Similar automation tool discussions
   - React SPA automation patterns

4. **Alternative Approaches**
   - Keyboard-only automation patterns
   - Accessibility tree interaction methods
   - Workarounds for React components
   - Alternative element targeting strategies

5. **Tool Comparison**
   - How other browsers handle React components
   - What makes Cursor IDE browser different
   - Feature gaps and limitations
   - Potential solutions from other tools

---

## EXPECTED RESEARCH OUTPUT

### Format Requested

1. **Executive Summary**
   - Are the tools broken or am I using them wrong?
   - Quick answer: What's the actual solution?

2. **Detailed Findings**
   - Research from each source
   - What works, what doesn't
   - Known issues and limitations

3. **Working Solution (If Exists)**
   - Complete step-by-step guide
   - Code examples that work
   - Verification methods

4. **If No Working Solution**
   - Confirmation tools are broken
   - Known bugs and issues
   - Workarounds or alternatives
   - Timeline for fixes

5. **Recommendations**
   - Immediate actions to take
   - Long-term solutions
   - Feature requests to submit

---

## CRITICAL INFORMATION SUMMARY

### What Works ✅
- Navigation (`browser_navigate`)
- Snapshots (`browser_snapshot`)
- Keyboard focus (`press_key` for "/")
- Waiting (`browser_wait_for`)
- Console/network access

### What Doesn't Work ❌
- Clicking (`browser_click` - script errors)
- Typing (`browser_type` - false positives)
- JavaScript execution (tool doesn't exist)

### What We Need 🔍
- Working method to type in React input components
- Working method to click buttons/toggles
- Element identification for custom components
- Verification methods
- OR confirmation that tools are broken

---

## TIMELINE & URGENCY

**Priority:** CRITICAL
**Urgency:** IMMEDIATE
**Blocking:** All automated research tasks for the project

**Project Impact:**
- Cannot perform deep research on Perplexity Academic
- Cannot automate research workflow
- Manual research required (inefficient)
- Project timeline at risk

---

## RESEARCHER INSTRUCTIONS

Please conduct comprehensive research to answer:

1. **Is there a working solution?** If yes, provide complete instructions.
2. **Are the tools broken?** If yes, confirm with evidence and provide alternatives.
3. **What's the correct usage pattern?** Provide tested, verified examples.
4. **Are there limitations?** Document what works and what doesn't.

**Sources to Check:**
- Official Cursor IDE documentation
- GitHub repositories and issues
- Community forums and discussions
- Similar tool documentation (for comparison)
- Any known workarounds or fixes

**Deliverable:** A comprehensive research report with actionable solutions or confirmation of tool limitations with alternatives.

---

**Research Request Date:** December 14, 2025
**Browser Tool Version:** Latest (2024-2025)
**Test Platform:** macOS (darwin 25.1.0)
**Cursor IDE:** Latest version

**Contact for Clarifications:** Provide findings that enable the AI assistant to successfully use the Cursor IDE built-in browser for web automation tasks.

---

## APPENDIX: Files Referenced

- `RESEARCH_PROMPT_BROWSER_USAGE.md` - Initial research prompt
- `BROWSER_TOOL_VERIFICATION.md` - Test results and evidence
- `BROWSER_LIMITATION_ANALYSIS.md` - Tool limitations analysis
- Snapshot files: `/Users/studio/.cursor/browser-logs/snapshot-*.log`

---

**END OF RESEARCH PROMPT**
