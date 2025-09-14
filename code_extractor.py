import ast
import os
import nbformat
from typing import List, Tuple, Dict, Any

from config import ANALYZABLE_EXTENSIONS


def extract_code_units(file_path):
    """Extract code units from various file types with enhanced parsing"""
    units = []
    
    if file_path.endswith(".py"):
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            code = f.read()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno
                    chunk = "\n".join(code.splitlines()[start_line-1:end_line])
                    metadata = {"file": file_path, "name": node.name, "type": type(node).__name__, "lines": end_line - start_line + 1}
                    units.append((chunk, metadata))
        except SyntaxError:
            metadata = {"file": file_path, "name": "unknown", "type": "file", "lines": len(code.splitlines())}
            units.append((code, metadata))
    
    elif file_path.endswith(".ipynb"):
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            notebook = nbformat.read(f, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                code = cell.source
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            chunk = code
                            metadata = {"file": file_path, "name": node.name, "type": type(node).__name__, "cell_index": notebook.cells.index(cell), "lines": len(code.splitlines())}
                            units.append((chunk, metadata))
                except SyntaxError:
                    metadata = {"file": file_path, "name": "unknown", "type": "cell", "cell_index": notebook.cells.index(cell), "lines": len(code.splitlines())}
                    units.append((code, metadata))
    
    elif file_path.endswith(ANALYZABLE_EXTENSIONS):
        try:
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                code = f.read()
            metadata = {"file": file_path, "name": os.path.basename(file_path), "type": "file", "lines": len(code.splitlines())}
            units.append((code, metadata))
        except Exception:
            pass  
    
    return units