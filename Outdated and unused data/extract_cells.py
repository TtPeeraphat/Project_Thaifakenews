import json
with open('test_4.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Check cells 12 and 13 (0-indexed)
for i in [12, 13]:
    print(f"\n{'='*60}")
    print(f"CELL {i+1}")
    print('='*60)
    cell = nb['cells'][i]
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    # Print first 1500 chars
    print(source[:1500])
    if len(source) > 1500:
        print(f"\n... (truncated, total length: {len(source)})")
