from egraph.pattern import ASTNode, Compiler
from egraph.egraph import EGraph, Node

def _rewrite_test(expected_matches, base_rows=2, base_op='mm', a_node_k=3):
    # AST Pattern
    a = ASTNode('?A', metadata={'rows': '?m', 'cols': '?k'})
    b = ASTNode('?B', metadata={'rows': '?k', 'cols': '?n'})
    base = ASTNode(base_op, children=[a, b], metadata={'rows': base_rows, 'cols': 5})
    
    # EGraph
    egraph = EGraph()
    a_node = Node('a', metadata={'rows': 2, 'cols': a_node_k})
    b_node = Node('b', metadata={'rows': 3, 'cols': 5})
    egraph.add(a_node)
    egraph.add(b_node)
    mm_node = Node('mm', (egraph.get_node_eclass_id(a_node), egraph.get_node_eclass_id(b_node)), metadata={'rows': 2, 'cols': 5})
    egraph.add(mm_node)
    
    # Compile
    comp = Compiler(egraph)
    comp.compile([base])
    
    comp.program.run()
    assert len(comp.program.matches) == expected_matches, comp.program.matches
    if len(comp.program.matches):
        return comp.program.matches[0]

def test_rewrite_basic():
    match = _rewrite_test(1)
    assert match['?m'] == 2
    assert match['?k'] == 3
    assert match['?n'] == 5

def test_rewrite_incorrect_1():
    _rewrite_test(0, a_node_k=4) # since a has k=4, b has k=3

def test_rewrite_incorrect_2():
    _rewrite_test(0, base_op='mm2')

def test_rewrite_incorrect_3():
    _rewrite_test(0, base_rows=3)
