from dataclasses import dataclass, field
from typing import Any, Dict, List

from scipy.cluster.hierarchy import DisjointSet

@dataclass(frozen=True) # TODO figure out which things are most important here
class Node:
    """
    Node for arithmetic operations ONLY right now

    I think egg directly uses the Language class as a node type
    """
    op: str
    children: tuple = field(default_factory=tuple) # list of eclass ids as the children. ORDER MATTERS!!

    def discriminant(self): # discriminant quickly lets us figure out if two things aren't equal
        return self.op

    def matches(self, other: "Node") -> bool:
        """
        Match only on operator and arity
        """
        return self.op == other.op and len(self.children) == len(other.children)

# eclass class
class EClass:
    """
    Represents an e-class, containing a set of nodes
    """
    id: int
    # nodes: set
    data: any
    # HERE: list of eclass ids of the parent enodes
    # Rust: original ids of parent enodes. not sure what the use is
    parents: set

    def __init__(self, cid, data=None, parents=None):
        self.id = cid

        # we literally don't need this for tracking nodes
        # self.nodes = set() if nodes is None else set(nodes)
        self.data = data

        # we can see that since enode.children was eclass ids, parents is tracked in here
        # for merge, rebuild etc.
        self.parents = set() if parents is None else set(parents)


# egraph with UMH
class EGraph:
    unionfind: DisjointSet # U(union find), store eclass_ids
    nodes: List[Node]

    # Canonical node to eclass id, seems like H(hashcons)?
    node2id: Dict[Node, int] # memo, seems to be just node2id

    pending: list # the worklists
    analysis_pending: list

    id2eclass: Dict[int, EClass] # eclass id to eclass object, M(map)?

    def __init__(self):
        self.unionfind = DisjointSet()
        self.curr_eclass_id = 1 # new eclass id
        # self.nodes = []
        self.node2id = dict()
        self.pending = []
        self.analysis_pending = []
        self.id2eclass = dict() # id2eclass
    
    def add_to_unionfind(self, eclass_id: int):
        assert eclass_id not in self.unionfind
        self.unionfind.add(eclass_id)
    
    def find(self, eclass_id: int):
        # Assumption: DisjoinSet.getitem should return the root node(the first added)
        # so we can use getitem to canonicalize
        if eclass_id not in self.unionfind:
            self.add_to_unionfind(eclass_id)
        return self.unionfind[eclass_id]
    
    def canonicalize(self, enode: Node):
        new_children = tuple(self.find(e) for e in enode.children)
        return Node(enode.op, new_children)
    
    def get_node_eclass_id(self, enode: Node):
        enode = self.canonicalize(enode)
        return self.node2id[enode]

    def new_singleton_eclass(self):
        eclass = EClass(self.curr_eclass_id)
        self.id2eclass[eclass.id] = eclass
        self.curr_eclass_id += 1
        return eclass.id
    
    def add(self, enode: Node):
        enode = self.canonicalize(enode)
        if enode in self.node2id: # hashcons(?)
            return self.node2id[enode]
        eclass_id = self.new_singleton_eclass()
        for child_eclass_id in enode.children:
            self.id2eclass[child_eclass_id].parents.add(eclass_id)
        self.unionfind.add(eclass_id)
        self.node2id[enode] = eclass_id
    
    def merge(self, eid1: int, eid2: int):
        """Merges two eclass ids"""
        # perform_union egraph.rs 1144
        # we can see that at 1177, we actually remove class2 and add class2 nodes to class1
        eid1, eid2 = (eid1, eid2) if len(self.id2eclass[eid1].parents) >= len(self.id2eclass[eid2].parents) else (eid2, eid1)
        merged = self.unionfind.merge(eid1, eid2) # I think eid1 will be the new root
        new_id = self.find(eid1)
        if not merged: # xy already merged
            return new_id
        
        # add class2 to new_id and delete
        # if class1 is not new_id, add class1 to new_id and delete too

        # We extend the worklist with the newly merged in nodes' parents
        # So in the paper, they say they extend with the class and look at parents.
        # here, they extend with parents and look at children I think
        self.pending.extend(self.id2eclass[eid2].parents)
        return new_id
    
    def process_unions(self):
        # egraph.rs 1333
        while len(self.pending):
            eclass_id = self.pending.pop()
            eclass: EClass = self.id2eclass[eclass_id]
            for i in range(len(eclass.nodes)):
                node = eclass.nodes[i]
                new_node = self.canonicalize(node)

                already_in_memo = not new_node.matches(node) and new_node in self.node2id
                
                # updates
                if already_in_memo:
                    self.merge(eclass_id, self.node2id[new_node])
                self.node2id.pop(node)
                self.node2id[new_node] = self.find(eclass_id)
                eclass.nodes[i] = new_node


# pattern matcher



# Extraction, I'm going to put the e-graph into a different format
class ExEClass:
    id: int
    nodes: List[Node]
    metadata: dict
    
    def __init__(self, cid, nodes):
        self.id = cid
        self.nodes = nodes
        self.metadata = dict()
    
    def add_node(self, n):
        self.nodes.append(n)
    
    def __repr__(self):
        return f'xClass({self.id})'
    
    def full_repr(self):
        return f'xClass(id={self.id}, nodes={[repr(n) for n in self.nodes]})'

class ExENode:
    op: str
    children: list # initially a list of int, and then we replace them all with e-classes
    def __init__(self, op, children):
        self.op = op
        self.children = children
        self.metadata = dict()
    
    def __repr__(self):
        assert all(isinstance(c, ExEClass) for c in self.children)
        return f'xNode({self.op}, {tuple(c.id for c in self.children)})'

def export_egraph(egraph: EGraph):
    eclasses = dict() # id to eclass
    x_enodes = []
    for n in egraph.node2id:
        # make find eclass
        eclass_id = egraph.node2id[n]
        repr_id = egraph.unionfind[eclass_id]
        if repr_id not in eclasses:
            eclasses[repr_id] = ExEClass(repr_id, [])
        x_eclass = eclasses[repr_id]
        x_enode = ExENode(n.op, n.children)
        x_eclass.add_node(x_enode)
        x_enodes.append(x_enode)
    
    for n in x_enodes:
        n.children = tuple(eclasses[egraph.unionfind[c]] for c in n.children)
    return eclasses, x_enodes


# TESTS
def test_node_hash():
    x = Node('+', (1, 2))
    y = Node('+', (1, 2))
    s = set([x, y])
    assert len(s) == 1


if __name__ == '__main__':
    egraph = EGraph()
    one = Node(1)
    two = Node(2)
    x = Node('x')
    egraph.add(one)
    egraph.add(two)
    egraph.add(x)
    plus = Node('+', (egraph.get_node_eclass_id(one), egraph.get_node_eclass_id(x)))
    plus2 = Node('+', (egraph.get_node_eclass_id(one), egraph.get_node_eclass_id(two)))
    egraph.add(plus)
    egraph.merge(egraph.get_node_eclass_id(x), egraph.get_node_eclass_id(two))
    egraph.process_unions()
    eclasses, x_enodes = export_egraph(egraph)
    print([ec.full_repr() for ec in eclasses.values()])
    print(x_enodes)
    # x = Node("+", [Node("1", []), Node("*", [Node("2", []), Node("3", [])])])
    # print(x)