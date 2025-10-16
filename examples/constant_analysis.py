from egraph.egraph import BasicAnalysis, Node, EGraph, EClass
from egraph.logging import debug

debug('EGRAPH')


def modify(obj: BasicAnalysis, egraph: EGraph, eclass_id, eclass: EClass) -> bool:
    if obj.eclass_map[eclass_id]:
        return False
    for n in eclass.nodes:
        n = egraph.canonicalize(n)
        if n.op == '*' and obj.eclass_map[n.children[0]] and obj.eclass_map[n.children[1]]:
            print(f'Making {eclass_id} constant')
            obj.eclass_map[eclass_id] = True
            return True
    return False

join_fn = lambda x, y: x or y
const_analysis = BasicAnalysis('const', False, join_fn, modify)
egraph = EGraph()
egraph.analyses.append(const_analysis)
# add x * 2 = y, and then add y * 3
x_id = egraph.add(Node('x'))
two_id = egraph.add(Node(2, metadata={'const': True}))
three_id = egraph.add(Node(3, metadata={'const': True}))
two_x_id = egraph.add(Node('*', (two_id, x_id)))
three_mult_id = egraph.add(Node('*', (two_x_id, three_id)))

# so now if we bind x with a constant, we should get that 2*x is constant and 3 * (2 * x) is constant
new_node_id = egraph.add(Node(1, metadata={'const': True}))
egraph.merge(new_node_id, x_id)
egraph.rebuild()
print(egraph.debug_str())
print(const_analysis.eclass_map)