
import json
import os

notebook_path = '/home/Ahmed-abobakr/Downloads/Da_ahmed/movie_analysis_pipeline.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

code_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Filter out jupyter magic commands
        source_lines = [line for line in source.split('\n') if not line.strip().startswith('!') and not line.strip().startswith('%')]
        code_cells.append("\n".join(source_lines))

full_script = "\n\n".join(code_cells)

# Add a check to ensure we are in the right dir
full_script = f"import os\nos.chdir('/home/Ahmed-abobakr/Downloads/Da_ahmed')\n\n{full_script}"

with open('/home/Ahmed-abobakr/Downloads/Da_ahmed/run_analysis.py', 'w') as f:
    f.write(full_script)

print("Created run_analysis.py")
