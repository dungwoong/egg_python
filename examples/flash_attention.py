from typing import List
from egraph.egraph import EGraph, Node, ExENode, ExEClass, export_egraph, extract_egraph_local_cost
from egraph.rewrite import Rewrite, ASTNode
from egraph.logging import silence_all, debug

silence_all()
debug('EGRAPH')


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

# let's just try a simple rewrite

# elem_div(x @ y) = elem_div(x) @ y
x_ast, y_ast = ASTNode('?x', metadata={'r': '?rx', 'c': '?cx'}), ASTNode('?y', metadata={'r': '?cx', 'c': '?cy'})
gemm_ast = ASTNode('@', [x_ast, y_ast], metadata={'r': '?rx', 'c': '?cy'})
before_root = ASTNode('elem_div', [gemm_ast], metadata={'r': '?rx', 'c': '?cy'})

after_div_ast = ASTNode('elem_div', [x_ast], metadata={'r': '?rx', 'c': '?cx'})
after_root = ASTNode('@', [after_div_ast, y_ast], metadata={'r': '?rx', 'c': '?cy'})
qk_rewrite = Rewrite.new([before_root], [after_root], egraph, 'qk')

# bdiv(x, vec) @ y = bdiv((x @ y), vec)
x_ast, vec_ast = ASTNode('?x', metadata={'r': '?rx', 'c': '?cx'}), ASTNode('?vec', metadata={'r': '?rx', 'c': 1}) 
bdiv_ast = ASTNode('b_div', [x_ast, vec_ast], metadata={'r': '?rx', 'c': '?cx'})
y_ast = ASTNode('?y', metadata={'r': '?cx', 'c': '?cy'})
before_root = ASTNode('@', [bdiv_ast, y_ast], metadata={'r': '?rx', 'c': '?cy'})

xygemm_ast = ASTNode('@', [x_ast, y_ast], metadata={'r': '?rx', 'c': '?cy'})
after_root = ASTNode('b_div', [xygemm_ast, vec_ast], metadata={'r': '?rx', 'c': '?cy'})
pv_rewrite = Rewrite.new([before_root], [after_root], egraph, 'pv')

rewrites = [qk_rewrite, pv_rewrite]
for i in range(3):
    for r in rewrites:
        r.reset() # NEED TO REMEMBER THIS PART
    for r in rewrites:
        r.find_rewrites()
    for r in rewrites:
        # print(f'applying {r.label}')
        r.apply_rewrites()
    egraph.rebuild()

print(egraph.debug_str())

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
                assert 'r' in n.metadata and 'c' in n.metadata, n
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

# Maybe we need to first make sure that simple rewrites where we can attach to children work
# Just in the pattern, we should be able to reuse nodes from the matcher inside the applier, and then
# and rewrite.py line 22 will do stuf.