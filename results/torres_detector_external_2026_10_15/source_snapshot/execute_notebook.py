"""Execute every notebook cell and export a readable HTML copy."""
from pathlib import Path
import json
import hashlib
import sys
import time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent
path = ROOT / "mosquito_wingbeat_estudo.ipynb"
notebook = nbformat.read(path, as_version=4)
begin = time.perf_counter()

def progress(cell, cell_index):
    title = cell.source.splitlines()[0][:80]
    print(f"Cell {cell_index + 1}/{len(notebook.cells)}: {title}", flush=True)

client = NotebookClient(notebook, timeout=1800, kernel_name="mosquito-wingbeat",
                        resources={"metadata": {"path": str(ROOT)}},
                        on_cell_start=progress, record_timing=True)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
errors = [output for cell in notebook.cells if cell.cell_type == "code"
          for output in cell.get("outputs", []) if output.output_type == "error"]
assert not errors
assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
exporter = HTMLExporter(template_name="lab")
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
html, _ = exporter.from_notebook_node(notebook)
(ROOT / "mosquito_wingbeat_estudo.html").write_text(html)
executor = Path(sys.executable)
executor_label = str(executor.relative_to(ROOT)) if executor.is_relative_to(ROOT) else executor.name
audit = dict(total_cells=len(notebook.cells),
             executed_code_cells=sum(cell.cell_type == "code" for cell in notebook.cells),
             error_outputs=len(errors), elapsed_seconds=time.perf_counter() - begin,
             executor_python=executor_label)
(ROOT / "results/notebook_execution.json").write_text(json.dumps(audit, indent=2))
digests = {}
for item in (ROOT / "results").rglob("*"):
    if item.is_file() and item.name != "sha256.json":
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digests[str(item.relative_to(ROOT))] = digest.hexdigest()
(ROOT / "results/sha256.json").write_text(json.dumps(digests, indent=2))
print(json.dumps(audit, indent=2), flush=True)
