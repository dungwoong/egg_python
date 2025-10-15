from egraph.egraph import EGraph, Node, export_egraph, extract_egraph_local_cost
from egraph.rewrite import Rewrite, ASTNode

egraph = EGraph()
x_id = egraph.add(Node('x'))
two_id = egraph.add(Node(2))
times_id = egraph.add(Node('*', (x_id, two_id)))
div_id = egraph.add(Node('/', (times_id, two_id)))

# x * 2 --> x << 1
before_root = ASTNode('*', [ASTNode('?x'), ASTNode(2)])
after_root = ASTNode('<<', [ASTNode('?x'), ASTNode(1)])
shift_rewrite = Rewrite.new([before_root], [after_root], egraph, "shift")

# (x * 2) / 2 = x * (2 / 2)
before_root = ASTNode('/', [ASTNode('*', [ASTNode('?x'), ASTNode('?n')]), ASTNode('?m')])
after_root = ASTNode('*', [ASTNode('?x'), ASTNode('/', [ASTNode('?n'), ASTNode('?m')])])
assoc_rewrite = Rewrite.new([before_root], [after_root], egraph, "assoc")

# x/x = 1
# we can just add extra binding checks for each var, should be okay?
before_root = ASTNode('/', [ASTNode('?x'), ASTNode('?x')])
after_root =  ASTNode(1)
div_rewrite = Rewrite.new([before_root], [after_root], egraph, "div")

# x * 1 = x
before_root = ASTNode('*', [ASTNode('?x'), ASTNode(1)])
after_root = ASTNode('?x')
mult_1_rewrite = Rewrite.new([before_root], [after_root], egraph, "mult1")

# a * b = b * a
before_root = ASTNode('*', [ASTNode('?x'), ASTNode('?y')])
after_root = ASTNode('*', [ASTNode('?y'), ASTNode('?x')])
commutative_rewrite = Rewrite.new([before_root], [after_root], egraph, 'comm')

# TODO we need to add conditional and dynamic rewrites
# TODO also, if we do equality saturation we'll try to apply the same rewrite multiple times too...
# like every single time we'll be trying the x <<1 rewrite. hmmm
rewrites = [shift_rewrite, assoc_rewrite, div_rewrite, mult_1_rewrite, commutative_rewrite]


for i in range(3):
    for r in rewrites:
        r.reset() # NEED TO REMEMBER THIS PART
    for r in rewrites:
        r.find_rewrites()
    for r in rewrites:
        print(f'applying {r.label}')
        r.apply_rewrites()
    egraph.process_unions()
    egraph.debug_print()

eclasses, _ = export_egraph(egraph)
# TODO in the future, probably don't wanna have to hardcode div_id in here
min_expr = extract_egraph_local_cost(eclasses.values(), eclasses[div_id], costs={'<<': 0.5})
print(f'Best expr: {min_expr}')