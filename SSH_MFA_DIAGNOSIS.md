# SSH MFA Prompt Missing - Diagnosis & Fix

## Problem Summary

When attempting `ssh hulk`, you expected to see a Duo MFA prompt, but instead got a hostname resolution error. The ControlMaster setup is interfering with proper authentication flow.

## Root Causes Identified

### Issue #1: SSH Config Host Alias Mismatch
**Current Config:**
```
Host hulk
    HostName hulk.alliancecan.ca
```

**Problem:** The `Host` entry only has one name (`hulk`), so:
- ✅ `ssh hulk` works (matches config)
- ❌ `ssh hulk.alliancecan.ca` doesn't work (no match in config, tries literal hostname resolution)

**Comparison with Working Clusters:**
```
Host nibi nibi.alliancecan.ca        # ← Has TWO names
    HostName nibi.alliancecan.ca
    ...

Host hulk                            # ← Has ONE name (WRONG!)
    HostName hulk.alliancecan.ca
    ...
```

### Issue #2: Missing SSH Identity Key
**Current Config:**
```
Host hulk
    # No IdentityFile specified!
```

**Problem:** Without specifying which key to use, SSH falls back to default key-trying order, which may not match the Alliance Canada key.

**Working Clusters:**
```
Host nibi nibi.alliancecan.ca
    HostName nibi.alliancecan.ca
    User amird
    IdentityFile ~/.ssh/alliance_automation    # ← Specified!
    ...
```

### Issue #3: Socket Directory Path Mismatch
**Current Config:**
```
ControlPath ~/.ssh/controlmasters/%r@%h:%p    # Wrong directory!
```

**Problem:** The socket directory was changed in recent documentation, but:
- Actual working sockets are in: `~/.ssh/sockets/`
- Config points to: `~/.ssh/controlmasters/`
- Result: ControlMaster tries to use non-existent socket, connection fails

**Evidence:**
```bash
$ ls ~/.ssh/sockets/
amird@fir.alliancecan.ca:22        ✅ Active socket
amird@nibi.alliancecan.ca:22       ✅ Active socket
narval                             ✅ Active socket

$ ls ~/.ssh/controlmasters/
# Empty directory
```

### Issue #4: No ControlPersist Setting Consistency
**Current Config:**
```
ControlPersist 10h        # Inconsistent (others use 48-72h)
```

**Problem:** Lower persistence window expires sooner, requiring re-authentication.

---

## Why MFA Prompt Isn't Appearing

The expected flow should be:

```
User runs: ssh hulk
    ↓
SSH reads config, finds: ControlMaster auto, ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ↓
ControlMaster checks: /Users/studio/.ssh/controlmasters/amird@hulk.alliancecan.ca:22
    ↓
Socket doesn't exist (wrong directory)
    ↓
SSH should: Create new connection → Trigger MFA
    ✅ MFA prompt should appear here
    ↓
Upon successful auth: Socket created, command executes
```

**What's Actually Happening:**

```
User runs: ssh hulk
    ↓
SSH reads config, applies hulk settings
    ↓
ControlMaster tries: /Users/studio/.ssh/controlmasters/ (empty, socket not found)
    ↓
Instead of creating new connection, SSH fails with hostname error
    ❌ MFA never triggers
    ✗ "Could not resolve hostname hulk.alliancecan.ca: nodename nor servname provided"
```

The error suggests ControlMaster is failing at socket creation, causing SSH to abort before attempting actual connection.

---

## Solution: Fix SSH Config

Update the `[[ Host hulk ]]` section in `~/.ssh/config`:

### Before (Broken):
```
Host hulk
    HostName hulk.alliancecan.ca
    User amird
    ControlMaster auto
    ControlPath ~/.ssh/controlmasters/%r@%h:%p
    ControlPersist 10h
    ServerAliveInterval 120
    ServerAliveCountMax 5
    StrictHostKeyChecking accept-new
    ConnectTimeout 30
    Compression yes
```

### After (Fixed):
```
Host hulk hulk.alliancecan.ca
    HostName hulk.alliancecan.ca
    User amird
    IdentityFile ~/.ssh/alliance_automation
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h:%p
    ControlPersist 48h
    ServerAliveInterval 120
    ServerAliveCountMax 5
    StrictHostKeyChecking accept-new
    ConnectTimeout 30
    Compression yes
```

**Changes Made:**
1. ✅ `Host hulk` → `Host hulk hulk.alliancecan.ca` (add FQDN alias)
2. ✅ Added: `IdentityFile ~/.ssh/alliance_automation` (specify alliance key)
3. ✅ `ControlPath ~/.ssh/controlmasters/` → `ControlPath ~/.ssh/sockets/` (use working directory)
4. ✅ `ControlPersist 10h` → `ControlPersist 48h` (match other clusters)

---

## Verify the Fix Works

After applying config changes:

### Test 1: Hostname Alias Works
```bash
ssh hulk 'hostname'
# Should work and show MFA if new connection, or use cached socket
```

### Test 2: FQDN Also Works
```bash
ssh hulk.alliancecan.ca 'hostname'
# Should work (now that HOST line includes FQDN alias)
```

### Test 3: MFA Prompt Appears (First Connection)
```bash
# Force new connection (remove cached socket)
rm ~/.ssh/sockets/amird@hulk.alliancecan.ca:22

# Now MFA should trigger
ssh hulk
# Should see: "Duo two-factor login for amird"
```

### Test 4: ControlMaster Socket Created
```bash
ls ~/.ssh/sockets/amird@hulk.alliancecan.ca:22
# Should show the socket file (srw-------)
```

### Test 5: Subsequent Connections Use Cache (No MFA)
```bash
ssh hulk 'echo "No MFA needed"'
# Should execute instantly without Duo prompt
```

---

## Testing the Fix Step by Step

### Step 1: Apply Config Changes
(See below - apply the replacements)

### Step 2: Clear Old Cache
```bash
rm ~/.ssh/controlmasters/* 2>/dev/null || true
# (Optional - old controlmasters directory won't interfere since config now points to sockets/)
```

### Step 3: Test Connection (Should Trigger MFA)
```bash
ssh -v hulk 'echo "Connection successful"'
# Watch for:
# - debug1: Connecting to hulk.alliancecan.ca 
# - "Duo two-factor login for amird" prompt
# - Enter Duo MFA on iPhone
```

### Step 4: Verify Socket Created
```bash
ls -la ~/.ssh/sockets/ | grep hulk
```

### Step 5: Test Cached Connection (No MFA)
```bash
time ssh hulk 'hostname'
# Should be instant (<1 second), no MFA prompt
```

---

## Important Notes

### Why This Matters
- **Without IdentityFile**: SSH tries all default keys, may not try `alliance_automation`, MFA fails
- **Wrong ControlPath**: ControlMaster creates socket in wrong location, cache never works
- **Missing Host Alias**: You can't use the FQDN if it's not in the Host line

### Persistence Windows
After these changes, your persistence windows match:
- **nibi**: 48 hours
- **fir**: 72 hours
- **narval**: 48 hours
- **hulk**: 48 hours (increased from 10h)

This means you authenticate once per 2-3 days, not once per 10 hours.

### Future Editing SSH Config
All Alliance Canada clusters should now follow same pattern:
- `Host ALIAS FQDN` (two names)
- `IdentityFile ~/.ssh/alliance_automation`
- `ControlPath ~/.ssh/sockets/%r@%h:%p` (consistent directory)
- `ControlPersist 48-72h` (reasonable window)

---

## What Success Looks Like

### First Connection to hulk:
```bash
$ ssh hulk
Duo two-factor login for amird

Enter a passcode or select one of the following options:

 1. Duo Push to Amir's iPhone (iOS)
 2. Phone call to Amir's iPhone (iOS)  
 3. SMS passcodes

Passcode or option (1-3): 1

[Push notification sent to iPhone; wait for approval...]

[amird@login.hulk ~]$
```

### Subsequent Connections (cached socket):
```bash
$ ssh hulk 'hostname'
hulk.alliancecan.ca

$ time ssh hulk 'nvidia-smi'
[...nvidia output...]
ssh hulk 'nvidia-smi'  0.00s user 0.00s system 0% cpu 0.897 total
```

No MFA prompts on subsequent connections for 48 hours!

---

## Apply the Fix Now

The SSH config corrections are ready to be applied (see next section).
