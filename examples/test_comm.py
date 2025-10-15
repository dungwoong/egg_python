from egraph.egraph import EGraph, Node, export_egraph, extract_egraph_local_cost
from egraph.rewrite import Rewrite, ASTNode

egraph = EGraph()
x_id = egraph.add(Node('x'))
two_id = egraph.add(Node(2))
times_id = egraph.add(Node('*', (x_id, two_id)))

before_root = ASTNode('*', [ASTNode('?x'), ASTNode('?y')])
after_root = ASTNode('*', [ASTNode('?y'), ASTNode('?x')])
commutative_rewrite = Rewrite.new([before_root], [after_root], egraph, 'comm')

egraph.debug_print()
for i in range(3):
    commutative_rewrite.reset()
    commutative_rewrite.find_rewrites()
    # print(commutative_rewrite.matcher.matches)
    commutative_rewrite.apply_rewrites()
    egraph.process_unions()
    egraph.debug_print()