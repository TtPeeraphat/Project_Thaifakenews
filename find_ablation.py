import json
with open('test_4.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Search for "AblationStudy" class definition in cells
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'class AblationStudy' in source:
        print(f"FOUND in CELL {i+1}\n")
        print(source[:2500])
        break
    if i >= 15:
        print(f"AblationStudy not found in first 16 cells")
        break
