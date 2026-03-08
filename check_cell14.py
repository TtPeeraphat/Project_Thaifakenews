import json
with open('test_4.ipynb', encoding='utf-8', errors='ignore') as f:
    nb = json.load(f)

# Check cells 13 and 14 (0-indexed)
for idx in [13, 14]:
    cell = nb['cells'][idx]
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    print(f"\n{'='*80}")
    print(f"CELL {idx+1}")
    print('='*80)
    # Find forward method
    if 'def forward' in source:
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'def forward' in line:
                # Print surrounding lines
                start = max(0, i-2)
                end = min(len(lines), i+15)
                print('\n'.join(lines[start:end]))
                break
    else:
        print(source[:800])
