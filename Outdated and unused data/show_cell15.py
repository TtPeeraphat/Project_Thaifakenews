import json
with open('test_4.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Get cell 15 (0-indexed = 14)
cell = nb['cells'][14]
source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
print(f"Cell 15 length: {len(source)} characters")
print(source)
