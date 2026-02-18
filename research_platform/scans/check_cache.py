import json
with open('/Users/studio/Desktop/PhD/Proposal/research_platform/scans/METADATA_CACHE.json', 'r') as f:
    cache = json.load(f)
total = len(cache)
with_created = len([m for m in cache.values() if m.get('created_at') != 'Unknown'])
print(f'Total: {total}')
print(f'With Created: {with_created}')
