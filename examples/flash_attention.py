from typing import List
from egraph.egraph import EGraph, Node, ExENode, ExEClass, export_egraph, extract_egraph_local_cost
from egraph.rewrite import Rewrite, ASTNode
from egraph.logging import silence_all

silence_all()


"""
Attention graph(simple)

Apply rewrites

Let's just see if we can model FA in our graph
"""

egraph = EGraph()
q_id = egraph.add(Node('Q', metadata={'r': 1024, 'c': 64}))
k_id = egraph.add(Node('Kt', metadata={'r': 64, 'c': 1024}))
qkgemm_id = egraph.add(Node('@', (q_id, k_id), metadata={'r': 1024, 'c': 1024}))
div_id = egraph.add(Node('elem_div', (qkgemm_id), metadata={'r': 1024, 'c': 1024}))
exp_id = egraph.add(Node('elem_exp', (div_id), metadata={'r': 1024, 'c': 1024}))
rowsum_id = egraph.add(Node('rowsum', (exp_id), metadata={'r': 1024, 'c': 1}))
# broadcast div
bdiv_id = egraph.add(Node('b_div', (exp_id, rowsum_id), metadata={'r': 1024, 'c': 1024}))
v_id = egraph.add(Node('V', metadata={'r': 1024, 'c': 64}))
pvgemm_id = egraph.add(Node('@', (bdiv_id, v_id), metadata={'r': 1024, 'c': 64}))


# Equality saturation here

# export
eclasses, _ = export_egraph(egraph)

def check_matrix_shapes(eclasses: List[ExEClass]):
    id2shape = dict() # eclass id to shape
    for eclass in eclasses:
        shape = None
        for n in eclass.nodes:
            if shape is None:
                shape = (n.metadata['r'], n.metadata['c'])
            else:
                assert (n.metadata['r'], n.metadata['c']) == shape
        id2shape[eclass.id] = shape
        eclass.metadata['shape'] = shape
    return id2shape

def _check_shapes(node: ExENode):
    match node.op:
        case '@':
            m, n = node['r'], node['c']
            k = node.children[0]['shape'][1] # child is an ExEClass
            if len(node.children) == 2 and node.children[0].metadata['shape'] == (m, k) and node.children[1].metadata['shape'] == (k, n):
                return True
        case 'elem_div' | 'elem_exp':
            if len(node.children) == 1 and node.children[0]['shape'] == (node['r'], node['c']):
                return True
        case 'rowsum':
            if len(node.children) == 1 and node.children[0]['shape'][0] == node['r']:
                return True
        case 'b_div':
            # matrix then vec
            if len(node.children) == 2 and node.children[0]['shape'][0] == node.children[1]['shape'][0] == node['r'] and node.children[0]['shape'][1] == node['c']:
                return True
        case _:
            return True
    return False

def check_shapes(eclasses: List[ExEClass]):
    good = True
    for eclass in eclasses:
        for enode in eclass.nodes:
            good = _check_shapes(enode) and good
    return good

print(check_matrix_shapes(eclasses.values()))
print(check_shapes(eclasses.values()))