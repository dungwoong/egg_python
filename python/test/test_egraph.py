from egraph.egraph import Node

def test_node_hash():
    x = Node('+', (1, 2))
    y = Node('+', (1, 2))
    s = set([x, y])
    assert len(s) == 1