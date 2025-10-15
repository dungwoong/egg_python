from egraph.egraph import *

def a_time_divide_example():
    egraph = EGraph(debug=True)
    # add (x * 2) / 2
    egraph.add(Node(2))
    egraph.add(Node('x'))
    times = Node('*', (egraph.get_node_eclass_id(Node(2)), egraph.get_node_eclass_id(Node('x'))))
    egraph.add(times)
    div = Node('/', (egraph.get_node_eclass_id(times), egraph.get_node_eclass_id(Node(2))))
    egraph.add(div)
    output_eclass_id = egraph.get_node_eclass_id(div)
    # add (x << 1) == (x * 2)
    egraph.add(Node(1))
    lshift = Node('<<', (egraph.get_node_eclass_id(Node('x')), egraph.get_node_eclass_id(Node(1))))
    egraph.add(lshift)
    egraph.merge(egraph.get_node_eclass_id(lshift), egraph.get_node_eclass_id(times))

    # (x * 2) / 2 = x * (2 / 2)
    print("(x * 2) / 2 = x * (2 / 2)")
    div2 = Node('/', (egraph.get_node_eclass_id(Node(2)), egraph.get_node_eclass_id(Node(2))))
    egraph.add(div2)
    times2 = Node('*', (egraph.get_node_eclass_id(Node('x')), egraph.get_node_eclass_id(div2)))
    egraph.add(times2)
    egraph.merge(egraph.get_node_eclass_id(div), egraph.get_node_eclass_id(times2))
    # 2 / 2 = 1
    print('2/2=1')
    egraph.merge(egraph.get_node_eclass_id(div2), egraph.get_node_eclass_id(Node(1)))
    egraph.process_unions()

    # x * 1 = x
    old = Node('*', (egraph.get_node_eclass_id(Node('x')), egraph.get_node_eclass_id(Node(1))))
    egraph.merge(egraph.get_node_eclass_id(old), egraph.get_node_eclass_id(Node('x')))
    egraph.process_unions()
    # print(egraph.node2id)

    eclasses, x_enodes = export_egraph(egraph)
    # print([ec.full_repr() for ec in eclasses.values()])
    min_expr = extract_egraph_local_cost(eclasses.values(), eclasses[output_eclass_id], costs={'<<': 0.5})
    print(f'Best expr: {min_expr}')

def hashable_dict_example():
    n = Node('*', metadata={'a': 1})
    n2 = Node('*', metadata={'a': 12})
    print(hash(n) == hash(n2))
    print(n2.metadata.get('c', 0))

if __name__ == '__main__':
    # hashable_dict_example()
    a_time_divide_example()