"""Execute read-only notebooks and persist BOTH their outputs and HTML."""
from pathlib import Path
import re
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parent


def main():
    for name in ['confiabilidade_binaria_aedes_2026_10_01','confiabilidade_binaria_aedes_adaptacao_2026_10_01']:
        path=ROOT/(name+'.ipynb');nb=nbformat.read(path,as_version=4)
        for cell in nb.cells:
            if cell.cell_type=='markdown':
                cell.source=re.sub(r'\]\(([^():]+\.md)\)',lambda m:'](reports/'+m[1]+')' if (ROOT/'reports'/m[1]).exists() and not (ROOT/m[1]).exists() else m[0],cell.source)
        NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
        if errors:raise RuntimeError('Notebook produced errors.')
        nbformat.write(nb,path);html,_=HTMLExporter().from_notebook_node(nb);(ROOT/(name+'.html')).write_text(html)
        print(name,'executed and saved,',sum(c.cell_type=='code' for c in nb.cells),'code cells, zero errors',flush=True)


if __name__=='__main__':main()
