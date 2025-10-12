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
    nodes: set
    data: any
    # HERE: list of eclass ids of the parent enodes
    # Rust: original ids of parent enodes. not sure what the use is
    parents: set

    def __init__(self, cid, data=None, nodes=None, parents=None):
        self.id = cid

        # we literally don't need this for tracking nodes
        self.nodes = set() if nodes is None else set(nodes)
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

    def new_singleton_eclass(self, node: Node):
        eclass = EClass(self.curr_eclass_id)
        eclass.nodes.add(node)
        self.id2eclass[eclass.id] = eclass
        self.curr_eclass_id += 1
        return eclass.id
    
    def add(self, enode: Node):
        enode = self.canonicalize(enode)
        if enode in self.node2id: # hashcons(?)
            return self.node2id[enode]
        eclass_id = self.new_singleton_eclass(enode)
        for child_eclass_id in enode.children:
            self.id2eclass[child_eclass_id].parents.add(eclass_id)
        self.unionfind.add(eclass_id)
        self.node2id[enode] = eclass_id
    
    def merge(self, eid1: int, eid2: int):
        """Merges two eclass ids"""
        # print(f'merging {eid1}, {eid2}')
        # perform_union egraph.rs 1144
        # we can see that at 1177, we actually remove class2 and add class2 nodes to class1
        eid1, eid2 = (eid1, eid2) if len(self.id2eclass[eid1].parents) >= len(self.id2eclass[eid2].parents) else (eid2, eid1)
        merged = self.unionfind.merge(eid1, eid2) # I think eid1 will be the new root
        new_id = self.find(eid1)
        if not merged: # xy already merged
            return new_id

        # maybe just canonicalize eid2 here?
        for n in self.id2eclass[eid2].nodes:
            # nothing should change with the nodes, so let's just update their hashcons entry
            self.node2id[self.canonicalize(n)] = new_id
        
        # NOTE in the egg source code, they self.classes.remove id2 and then they do concat_vecs on the nodes and parents
        # SO WHAT TF IS THE POINT OF THE UNION FIND THEN???
        self.id2eclass[new_id].nodes = self.id2eclass[new_id].nodes.union(self.id2eclass[eid2].nodes)
        self.id2eclass[new_id].parents = self.id2eclass[new_id].parents.union(self.id2eclass[eid2].parents)

        
        # add class2 to new_id and delete
        # if class1 is not new_id, add class1 to new_id and delete too

        # We extend the worklist with the newly merged in nodes' parents
        # So in the paper, they say they extend with the class and look at parents.
        # here, they extend with parents and look at children I think
        self.pending.extend(self.id2eclass[eid2].parents)
        # since we merged, we also have to add eid1's parents
        self.pending.extend(self.id2eclass[eid1].parents)
        return new_id
    
    def process_unions(self):
        # egraph.rs 1333
        while len(self.pending):
            eclass_id = self.pending.pop()
            eclass: EClass = self.id2eclass[eclass_id]
            nodes = list(eclass.nodes)
            # print(f'repairing {eclass_id} {nodes}')
            for node in nodes:
                new_node = self.canonicalize(node)
                # print(f'{node} -> {new_node}')

                already_in_memo = not new_node.matches(node) and new_node in self.node2id
                
                # updates
                if already_in_memo:
                    self.merge(eclass_id, self.node2id[new_node])
                if node in self.node2id:
                    self.node2id.pop(node)
                self.node2id[new_node] = self.find(eclass_id)
                eclass.nodes.remove(node)
                eclass.nodes.add(new_node)


# pattern matcher



# Extraction, I'm going to put the e-graph into a different format
class ExENode:
    op: str
    children: list # initially a list of int, and then we replace them all with e-classes
    def __init__(self, op, children):
        self.op = op
        self.children = children
        self.metadata = dict()
    
    def __repr__(self):
        assert all(isinstance(c, ExEClass) for c in self.children)
        return f'xNode({self.op}, {tuple(c.id for c in self.children)})' if self.children else f'xNode({self.op})'

class ExtractedEnode:
    """
    After extraction process
    """
    op: str
    children: list
    def __init__(self, op, children=None):
        self.op = op
        self.children = children if children is not None else []
    
    @staticmethod
    def from_enode(enode: ExENode):
        return ExtractedEnode(enode.op)
    
    def __repr__(self):
        args = ' '.join([str(c) for c in self.children]) if self.children else ''
        return f'({self.op} {args})' if self.children else str(self.op)


class ExEClass:
    id: int
    nodes: List[ExENode]
    metadata: dict
    
    def __init__(self, cid, nodes):
        self.id = cid
        self.nodes = nodes
        self.metadata = dict()
    
    def add_node(self, n):
        self.nodes.append(n)
    
    def get_children(self):
        children = set()
        for n in self.nodes:
            children = children.union(n.children)
        return children
    
    def __repr__(self):
        return f'xClass({self.id})'
    
    def full_repr(self):
        return f'xClass(id={self.id}, nodes={[repr(n) for n in self.nodes]})'

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

# def topology_sort(classes: List[ExEClass], parents_first=True):
#     stack = list(classes)
#     output_list = []
#     started = dict()
#     finished = dict()
#     time = 0
#     while len(stack):
#         time += 1
#         curr = stack.pop()
#         if curr in finished:
#             continue
#         if curr in started:
#             finished[curr] = time
#             continue
#         output_list.append(curr)
#         started[curr] = time
#         stack.append(curr)
#         children = curr.get_children()
#         for child in children:
#             stack.append(child)
#     output_list = list(sorted(output_list, key=lambda n: finished[n], reverse=parents_first))
#     return output_list

def extract_egraph_local_cost(eclasses: List[ExEClass], class_to_extract, costs, default_cost=1):
    # once we support pattern matching, we don't need to add class_to_extract now
    for eclass in eclasses:
        eclass.metadata['cost'] = 1e9
        eclass.metadata['argmin'] = None
    
    def get_children_cost(enode: ExENode):
        cost = 0
        for ecl in enode.children:
            cost += ecl.metadata['cost']
        return cost
    
    keep_going = True
    while keep_going:
        keep_going = False
        for eclass in eclasses:
            for enode in eclass.nodes:
                new_cost = get_children_cost(enode) + costs.get(enode.op, default_cost)
                if new_cost != enode.metadata.get('cost', None):
                    keep_going = True
                    enode.metadata['cost'] = new_cost
                if enode.metadata['cost'] < eclass.metadata['cost']:
                    eclass.metadata['cost'] = enode.metadata['cost']
                    eclass.metadata['argmin'] = enode
    # Extract
    def extract(eclass: ExEClass):
        min_enode = eclass.metadata['argmin']
        extracted = ExtractedEnode.from_enode(min_enode)
        for c in min_enode.children:
            extracted.children.append(extract(c))
        return extracted
    return extract(class_to_extract)


# TESTS
def test_node_hash():
    x = Node('+', (1, 2))
    y = Node('+', (1, 2))
    s = set([x, y])
    assert len(s) == 1

def test_basic_example():
    egraph = EGraph()
    one = Node(1)
    two = Node(2)
    x = Node('x')
    egraph.add(one)
    egraph.add(two)
    egraph.add(x)
    one1, two1 = Node(1), Node(2)
    egraph.add(one1)
    egraph.add(two1)
    plus = Node('+', (egraph.get_node_eclass_id(one), egraph.get_node_eclass_id(x)))
    plus2 = Node('+', (egraph.get_node_eclass_id(one1), egraph.get_node_eclass_id(two1)))
    egraph.add(plus)
    egraph.merge(egraph.get_node_eclass_id(x), egraph.get_node_eclass_id(two))
    egraph.process_unions()
    eclasses, x_enodes = export_egraph(egraph)
    print([ec.full_repr() for ec in eclasses.values()])
    print(x_enodes)


if __name__ == '__main__':
    egraph = EGraph()
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

# Observations
# 1. ok like why are we just holding a bunch of nodes that we're not using?
# when we merge, we merge class1.nodes, class2.nodes. Do we even need a union find then??
# 2. you might not actually have some form of nodes in node2id, so we added
# if node in self.node2id: self.node2id.pop(node) to process_unions. not sure when this would happen...
# 3. extraction process is...interesting, because you might have e.g. x * 0 = 0, x*0 and 0 are in the same class so you are recursing on yourself

# next steps:
# - change the format of the graph, this is kinda dumb
# - add analysis for joins and add that to rebuilding
# - add the rewrite VM