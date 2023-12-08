from goatools import obo_parser
import requests

# Load the GO OBO file
go = obo_parser.GODag('/local2/yuyan/gene_emb/data/benchmark/GO/go-basic.obo')


def trace_to_root_level(go_id, go, level=1):
    """
    Trace back the GO term to specified level (default is one level under the root)
    """
    term = go[go_id]
    parents = list(term._parents)[0]
    
    while term.level > level:
        if not parents:
            break
        term = go[parents]
        parents = list(term._parents)[0]

    return term.id

BP_id = 'GO:0008150'
BP_term = go[BP_id]
BP_childs = list(BP_term.children)
BP_childs = [ch.id for ch in BP_childs]

