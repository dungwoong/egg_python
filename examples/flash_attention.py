from typing import List
from egraph.egraph import EGraph, Node, ExENode, ExEClass, export_egraph, extract_egraph_local_cost, remove_nodes_above_min_cost
from egraph.rewrite import Rewrite, ASTNode
from egraph.logging import silence_all, debug
from egraph.util import topology_sort

silence_all()
debug('EGRAPH')


"""
Attention graph(simple)

Apply rewrites

Let's just see if we can model FA in our graph
"""
# ###################################################################
# Initial graph with metadata
# ###################################################################

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

# ###################################################################
# Rewrites
# ###################################################################

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


# ###################################################################
# Apply rewrites in equality saturation
# ###################################################################

# TODO in the future, we should check for saturation and have a stopping rule
rewrites = [qk_rewrite, pv_rewrite]
for i in range(10):
    for r in rewrites:
        r.reset()
    for r in rewrites:
        r.find_rewrites()
    for r in rewrites:
        r.apply_rewrites()
    egraph.rebuild()

print(egraph.debug_str())


# ###################################################################
# Export e-graph to graph format
# ###################################################################
eclasses, _ = export_egraph(egraph)
exported_eclasses = list(eclasses.values())
output_eclass = eclasses[egraph.find(pvgemm_id)] # TODO better lookup functions


# ###################################################################
# Apply some checks
# ###################################################################
def check_matrix_shapes(eclasses: List[ExEClass]):
    """Check each e-class has one unique shape"""
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
    """Check that each node's input shapes make sense"""
    good = True
    for eclass in eclasses:
        for enode in eclass.nodes:
            good = _check_shapes(enode) and good
    return good

class_shapes = check_matrix_shapes(eclasses.values())
if check_shapes(eclasses.values()):
    print('Checks passed :)')


# ###################################################################
# Custom FLOP cost function
# ###################################################################

def flop_cost_fn(node: ExENode):
    """Get FLOPs for a specific operator"""
    out_r, out_c = (node['r'], node['c'])
    match node.op:
        case '@':
            m, n = node['r'], node['c']
            k = node.children[0]['shape'][1] # child is an ExEClass
            return 2 * m * n * k
        case 'elem_div' | 'elem_exp':
            return out_r * out_c
        case 'rowsum':
            return out_r * node.children[0]['shape'][1]
        case 'b_div':
            return out_r * out_c
        case _:
            return 0

def get_flop_score(eclasses: List[ExEClass]):
    """
    Iterate over children and calculate FLOP score until
    we converge on minimum scores for each e-class
    """
    # once we support pattern matching, we don't need to add class_to_extract now
    eclasses = topology_sort(eclasses, lambda x: x.get_children(), parents_first=False)

    for eclass in eclasses:
        eclass.metadata['cost'] = float('inf')
        eclass.metadata['argmin'] = None
        eclass.metadata['cost_breakdown'] = {'placeholder': float('inf')}
    
    def get_children_cost(enode: ExENode):
        enode.metadata['cost_breakdown'] = {enode: flop_cost_fn(enode)}
        for ecl in enode.children:
            enode.metadata['cost_breakdown'].update(ecl.metadata['cost_breakdown'])
        
        return sum(enode.metadata['cost_breakdown'].values())
    
    keep_going = True
    while keep_going:
        keep_going = False
        for eclass in eclasses:
            for enode in eclass.nodes:
                new_cost = get_children_cost(enode)
                if new_cost != enode.metadata.get('cost', None):
                    keep_going = True
                    enode.metadata['cost'] = new_cost
                if enode.metadata['cost'] < eclass.metadata['cost']:
                    eclass.metadata['cost'] = enode.metadata['cost']
                    eclass.metadata['argmin'] = enode
                    eclass.metadata['cost_breakdown'] = enode.metadata['cost_breakdown']

get_flop_score(exported_eclasses)
# remove_nodes_above_min_cost(exported_eclasses)

for c in exported_eclasses:
    print(c.full_repr(ignore_metadata=set(['cost_breakdown'])))

# SAMPLE OUTPUT
# xClass(id=1, nodes=['Q'])
# xClass(id=2, nodes=['Kt'])
# xClass(id=3, nodes=["(@ 1 2 {'r': 1024, 'c': 1024, 'cost': 134217728})"])
# xClass(id=4, nodes=["(elem_div 3 {'r': 1024, 'c': 1024, 'cost': 135266304})", "(@ 10 2 {'r': 1024, 'c': 1024, 'cost': 134283264})"])
# xClass(id=5, nodes=["(elem_exp 4 {'r': 1024, 'c': 1024, 'cost': 135331840})"])
# xClass(id=6, nodes=["(rowsum 5 {'r': 1024, 'c': 1, 'cost': 136380416})"])
# xClass(id=7, nodes=["(b_div 5 6 {'r': 1024, 'c': 1024, 'cost': 137428992})"])
# xClass(id=8, nodes=['V'])
# xClass(id=9, nodes=["(@ 7 8 {'r': 1024, 'c': 64, 'cost': 271646720})", "(b_div 12 6 {'r': 1024, 'c': 64, 'cost': 270663680})"])
# xClass(id=10, nodes=["(elem_div 1 {'r': 1024, 'c': 64, 'cost': 65536})"])
# xClass(id=12, nodes=["(@ 5 8 {'r': 1024, 'c': 64, 'cost': 269549568})"])