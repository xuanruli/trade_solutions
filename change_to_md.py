import json
import re

with open('pairs.ipynb', 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

readme_content = ""


for cell in notebook_content['cells']:
    cell_type = cell['cell_type']
    source = ''.join(cell['source'])
    
    if cell_type == 'markdown':
        
        readme_content += source + "\n\n"
    elif cell_type == 'code':
        
        if source.strip():  
            readme_content += "```python\n" + source + "\n```\n\n"
  


with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("Conversion completed! README.md has been created.")