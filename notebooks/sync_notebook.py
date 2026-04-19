import json
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parent.parent
PY_FILE = ROOT_DIR / "notebooks" / "01_eda.py"
NB_FILE = ROOT_DIR / "notebooks" / "01_eda_and_hypothesis_testing.ipynb"

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }

def create_code_cell(source):
    # Ensure matplotlib inline for notebook
    if "import matplotlib" in source and "%matplotlib inline" not in source:
        source = "%matplotlib inline\n" + source.replace('matplotlib.use("Agg")', '')
    
    # Replace __file__ logic which fails in notebooks
    source = source.replace('ROOT_DIR    = Path(__file__).resolve().parent.parent', 'ROOT_DIR    = Path.cwd().parent')
        
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }

def convert_py_to_ipynb():
    content = PY_FILE.read_text(encoding="utf-8")
    
    # Split by the header delimiters
    blocks = re.split(r'# ══════════════════════════════════════════════\n', content)
    
    cells = []
    
    # Block 0 contains the docstring and imports
    header_block = blocks[0].strip()
    
    # Extract the top docstring for a markdown cell
    docstring_match = re.search(r'"""(.*?)"""', header_block, re.DOTALL)
    if docstring_match:
        cells.append(create_markdown_cell(docstring_match.group(1).strip()))
        # Remove docstring from code
        header_block = header_block.replace(docstring_match.group(0), "").strip()
        
    cells.append(create_code_cell(header_block))
    
    # Iterate through remaining blocks
    for block in blocks[1:]:
        if not block.strip():
            continue
            
        # The first non-empty line usually looks like: # FIGURE X — Return distributions
        lines = block.split("\n")
        title_line = ""
        code_lines = []
        
        for line in lines:
            if line.startswith("# FIGURE") or line.startswith("# Main") or line.startswith("# ───"):
                title_line = line.replace("# ", "").strip()
            else:
                code_lines.append(line)
                
        if title_line:
            cells.append(create_markdown_cell(f"### {title_line}"))
            
        code = "\n".join(code_lines).strip()
        if code:
            # If it contains the entry point, split the definition from the execution call
            if "if __name__ == \"__main__\":" in code:
                # 1. Remove the if __name__ block to leave the definition
                definition = re.sub(r'if __name__ == "__main__":.*', '', code, flags=re.DOTALL).strip()
                if definition:
                    cells.append(create_code_cell(definition))
                
                # 2. Add a clear header and the standalone final call
                cells.append(create_markdown_cell("### Execution Interface\nRun the integrated EDA pipeline now."))
                cells.append(create_code_cell("run_eda_pipeline()"))
            else:
                cells.append(create_code_cell(code))
            
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    NB_FILE.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Successfully synced {PY_FILE.name} into {NB_FILE.name} via standard JSON.")

if __name__ == "__main__":
    convert_py_to_ipynb()
