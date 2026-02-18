# Research Prompt: How to Correctly Use Cursor IDE Built-in Browser for Web Automation

## Context & Background

I am an AI assistant (Auto) operating within Cursor IDE. I need to conduct automated research using the built-in Cursor IDE browser (not the MCP Playwright browser) to interact with Perplexity Academic and perform deep research queries.

## Current Situation

### What I'm Trying to Do
1. Navigate to Perplexity Academic (`https://www.perplexity.ai/academic`)
2. Enable Deep Research mode (circuit board icon with Q)
3. Enable Academic Sources toggle (in "More sources" dropdown menu)
4. Type and submit research queries about Mamba-2, BiMamba, genomics, CRISPR prediction
5. Capture and extract research results for analysis

### Tools Available

I have access to these Cursor IDE built-in browser tools:
- `mcp_cursor-ide-browser_browser_navigate(url)` - Navigation works ✅
- `mcp_cursor-ide-browser_browser_snapshot()` - Captures page state ✅
- `mcp_cursor-ide-browser_browser_click(element, ref)` - **FAILING** ❌
- `mcp_cursor-ide-browser_browser_type(element, ref, text, submit?)` - **REPORTS SUCCESS BUT NOTHING APPEARS** ❌
- `mcp_cursor-ide-browser_browser_press_key(key)` - Works for navigation keys ✅
- `mcp_cursor-ide-browser_browser_wait_for(time, text?, textGone?)` - Works ✅
- `mcp_cursor-ide-browser_browser_hover(element, ref)` - Not tested
- `mcp_cursor-ide-browser_browser_select_option(element, ref, values)` - Not tested
- `mcp_cursor-ide-browser_browser_resize(width, height)` - Not needed

## Specific Problems Encountered

### Problem 1: Click Operations Fail
**Attempt:**
```javascript
mcp_cursor-ide-browser_browser_click(element="Deep Research icon", ref="ref-cursor-el-182")
```
**Result:**
```
Error executing tool browser_click: Script failed to execute, this normally means an error was thrown.
```

**What I've Tried:**
- Using ref from snapshot
- Using different ref formats
- Using element names from accessibility tree
- Using keyboard navigation (Tab key works)

### Problem 2: Type Operations Report Success But Don't Work
**Attempt:**
```javascript
mcp_cursor-ide-browser_browser_press_key(key="/")  // Focus search - WORKS
mcp_cursor-ide-browser_browser_type(element="Search input", ref="ref-m3nlh9iibv", text="Mamba-2 BiMamba...", submit=true)
```
**Result:**
```
- Characters typed: 41
- Form submitted: yes (Enter key pressed)
```
**BUT:** Snapshot shows NO CHANGE - page remains on default Academic homepage, no query visible, no results

**Verification:**
- Compared snapshots before/after: Identical (same file size, same structure)
- No query text appears in accessibility tree
- URL doesn't change
- No search results appear

### Problem 3: Cannot Find Correct Input Element

**What I've Searched For:**
- `<input>` elements - NOT FOUND in accessibility snapshot
- `<textarea>` elements - NOT FOUND
- `role="textbox"` - NOT FOUND
- `contenteditable` attributes - NOT FOUND
- Generic elements with text input capability - UNCLEAR

**What I Found:**
- Snapshot contains only `role: generic`, `role: button`, `role: link`, `role: img`
- Search input appears to be a custom React component
- Actual DOM path from user: `<p dir="auto"></p>` inside nested divs
- Typeahead menu (`role: listbox`) appears but no input element visible

## Technical Details

### Browser Environment
- **Browser Type:** Cursor IDE Built-in Browser (Chromium-based based on MCP cursor-ide-browser)
- **Page:** Perplexity Academic (React SPA)
- **Accessibility API:** Using ARIA/accessibility tree snapshots
- **Element References:** Using `ref` values from snapshot (e.g., `ref-m3nlh9iibv`)

### Page Structure Analysis
```
Perplexity Academic Page:
- Uses React/Virtual DOM
- Custom search input component (not standard HTML input)
- Dynamic element references (change on each snapshot)
- Complex nested div structure
- Accessibility tree shows generic roles, not semantic input elements
```

### Successful Operations
✅ Navigation: `browser_navigate()` works perfectly
✅ Snapshots: `browser_snapshot()` captures full page state
✅ Keyboard: `press_key("/")` successfully focuses search (based on behavior)
✅ Waiting: `wait_for()` works correctly

### Failed Operations
❌ `browser_click()` - Script execution errors
❌ `browser_type()` - Reports success but nothing appears on page
❌ Finding correct input element type/ref

## What I Need to Know

### Primary Research Questions

1. **How do I correctly identify and target editable text input elements in Cursor IDE built-in browser when:**
   - The accessibility tree shows only `role: generic` (no `role: textbox`)
   - The element is a custom React component (not standard HTML input/textarea)
   - The actual DOM shows `<p dir="auto">` inside nested divs

2. **Why does `browser_type()` report success but no text appears?**
   - Tool reports "Characters typed: 41" and "Form submitted: yes"
   - But page snapshots show no changes
   - No query text visible in accessibility tree
   - Is there a validation step I'm missing?

3. **How do I correctly use element references (`ref`) from snapshots?**
   - Should I use the exact ref string (e.g., `ref-m3nlh9iibv`)?
   - Or should I use a different format?
   - Do refs need to be stable or do they change on each snapshot?

4. **For `browser_click()` failures:**
   - What causes "Script failed to execute" errors?
   - How do I correctly target clickable elements (buttons, toggles, switches)?
   - Are there specific element types or attributes required?

5. **Alternative approaches:**
   - Should I use keyboard navigation entirely (Tab, Space, Enter)?
   - Are there specific keyboard shortcuts for Perplexity?
   - Can I interact with elements differently?

### Specific Use Cases Needed

**Use Case 1: Clicking Toggle Switches**
- Target: Toggle switch for "Academic Sources"
- HTML: `<button type="button" role="switch" aria-checked="false" data-state="unchecked">`
- Current ref from snapshot: Various (changes per snapshot)
- Need: Working method to toggle switches

**Use Case 2: Clicking Mode Buttons**
- Target: Deep Research mode button (circuit board icon with Q)
- React Component: `IconPerplexityDeepResearch`
- Current ref: Changes per snapshot
- Need: Working method to activate mode buttons

**Use Case 3: Typing in Search Field**
- Target: Search input (appears as `<p dir="auto">` in DOM)
- Accessibility: Shows as nested `generic` roles
- Current approach: Using ref from parent generic element
- Need: Correct element to target and method that actually works

**Use Case 4: Submitting Queries**
- Current: Using `submit=true` parameter
- Result: Reports submission but nothing happens
- Need: Verification method and correct submission approach

## Requested Research Approach

Please research:

1. **Official Documentation**
   - Cursor IDE browser automation documentation
   - MCP cursor-ide-browser specification
   - Element selection and interaction patterns

2. **Technical Forums/Issues**
   - GitHub issues for cursor-ide-browser
   - Stack Overflow questions about Cursor IDE browser automation
   - Community discussions about similar problems

3. **Best Practices**
   - How others interact with React SPAs using accessibility trees
   - Patterns for contenteditable/custom input elements
   - Workarounds for elements without semantic roles

4. **Debugging Methods**
   - How to verify if browser_type actually worked
   - How to inspect what element is actually being targeted
   - Error logging and diagnostic approaches

5. **Alternative Solutions**
   - Keyboard-only navigation patterns
   - Different element targeting strategies
   - Workarounds for custom React components

## Expected Deliverable

A comprehensive guide that answers:
1. ✅ Step-by-step instructions for correctly using Cursor IDE built-in browser
2. ✅ How to identify and target custom React input elements
3. ✅ Why browser_type reports success but doesn't work
4. ✅ How to successfully click toggle switches and buttons
5. ✅ Verification methods to confirm actions actually occurred
6. ✅ Code examples showing correct usage patterns
7. ✅ Troubleshooting guide for common issues
8. ✅ Any limitations or known bugs with the browser tools

## Critical Requirements

- Must work with **Cursor IDE built-in browser** (NOT MCP Playwright browser)
- Must handle **React SPA** with dynamic content
- Must work with **custom input components** (not standard HTML)
- Must provide **verification methods** to confirm actions
- Must include **working examples** for the specific Perplexity Academic interface

---

**Research Priority:** CRITICAL - Blocking automated research workflow
**Urgency:** HIGH - Needed immediately to proceed with project
**Tool Version:** Latest Cursor IDE built-in browser (2024-2025)

---

## Additional Context

### What Works in Similar Tools

In MCP Playwright browser (which I'm NOT supposed to use):
- `mcp_playwright_browser_snapshot()` - Works, shows full accessibility tree
- `mcp_playwright_browser_type()` - Works, requires element and ref from snapshot
- `mcp_playwright_browser_click()` - Works, requires element and ref from snapshot

**Question:** Are the Cursor IDE browser tools supposed to work similarly? Why are they failing?

### User Instructions

User explicitly stated:
- "always only use your built-in browser and not the MCP browser unless I tell you otherwise"
- "You must figure out how to work with your built-in browser correctly"

This means I cannot switch to MCP Playwright browser - I must make the built-in browser work.

### Snapshot Analysis Example

From a typical snapshot:
```yaml
- role: generic
  ref: ref-m3nlh9iibv
  children:
    - role: generic
      ref: ref-q69dotd4i18
      children:
        # ... nested structure
        # No role: textbox anywhere
        # No input/textarea elements
```

**Question:** Which ref should I use for typing? The parent generic? A child? How do I know?

---

**Please provide comprehensive research findings that will enable me to correctly use the Cursor IDE built-in browser for web automation tasks.**
