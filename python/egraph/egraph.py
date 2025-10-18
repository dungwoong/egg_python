from dataclasses import dataclass, field
from typing import Any, Dict, List
from egraph.util import Hashabledict

from scipy.cluster.hierarchy import DisjointSet

from egraph.logging import egraph_logger as logger

@dataclass(frozen=True) # TODO figure out which things are most important here
class Node:
    """
    Node for arithmetic operations ONLY right now

    I think egg directly uses the Language class as a node type
    """
    op: str
    children: tuple = field(default_factory=tuple) # list of eclass ids as the children. ORDER MATTERS!!
    metadata: Hashabledict = field(default_factory=Hashabledict)

    def __post_init__(self):
        if not isinstance(self.metadata, Hashabledict):
            object.__setattr__(self, "metadata", Hashabledict(self.metadata))
        if not isinstance(self.children, tuple):
            if isinstance(self.children, int): # single id
                object.__setattr__(self, "children", (self.children,))
            else:
                object.__setattr__(self, "children", tuple(self.children))

    def discriminant(self): # discriminant quickly lets us figure out if two things aren't equal
        return self.op

    def matches(self, other) -> bool:
        """
        Match only on operator and arity
        """
        return self.op == other.op and len(self.children) == len(other.children)

    def __repr__(self):
        return f'N({self.op}{", " if self.children else ""}{self.children if self.children else ""})'


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
    
    def __repr__(self):
        nodes = ', '.join(str(n) for n in self.nodes)
        return f'C{self.id}({nodes})'


class EClassAnalysis:
    """
    Analysis is tied to some metadata field on a node
    wait a sec, ok with constant folding, can't we literally just...write them as transforms...
    """
    def __init__(self):
        self.eclass_map = dict() # eclassid : analysis data

    def make(self, enode: Node, eclass_id: int):
        # saves analysis data to the map
        raise NotImplementedError()

    def join(self, new_id: int, old_ids: list) -> bool:
        # for each id in old_ids, join it with new_id and update records
        # return whether the new data is different than the old data
        raise NotImplementedError()
    
    # modify can just be a rewrite. Let's think about that later.
    def modify(self, egraph, eclass_id, eclass: EClass) -> bool:
        # return whether this eclasses analysis changed
        pass

    def recreate_analysis(self, eclass_id, eclass: EClass) -> bool:
        # return whether this eclasses analysis changed
        pass

    def remove(self, eclass_id):
        if eclass_id in self.eclass_map:
            self.eclass_map.pop(eclass_id)

class BasicAnalysis(EClassAnalysis):
    def __init__(self, metadata_field, default_data_value, join_fn, modify):
        super().__init__()
        self.field = metadata_field
        self.default = default_data_value
        self.join_fn = join_fn
        self.modify_impl = modify
    
    def make(self, enode: Node, eclass_id: int):
        self.eclass_map[eclass_id] = enode.metadata.get(self.field, self.default)
    
    def join(self, new_id: int, old_ids: list) -> bool:
        old_data = self.eclass_map[new_id]
        for old_id in old_ids:
            self.eclass_map[new_id] = self.join_fn(self.eclass_map[new_id], self.eclass_map[old_id])
        return old_data != self.eclass_map[new_id]

    def recreate_analysis(self, eclass_id, eclass):
        old_data = self.eclass_map[eclass_id]
        for n in eclass.nodes:
            self.eclass_map[eclass_id] = self.join_fn(self.eclass_map[eclass_id], n.metadata.get(self.field, self.default))
        return old_data != self.eclass_map[eclass_id]
    
    def modify(self, egraph, eclass_id, eclass: EClass) -> bool:
        return self.modify_impl(self, egraph, eclass_id, eclass)


# egraph with UMH
class EGraph:
    unionfind: DisjointSet # U(union find), store eclass_ids
    nodes: List[Node]

    # Canonical node to eclass id, seems like H(hashcons)?
    node2id: Dict[Node, int] # memo, seems to be just node2id

    pending: list # the worklists
    analysis_pending: list

    analyses: List[EClassAnalysis]

    id2eclass: Dict[int, EClass] # eclass id to eclass object, M(map)?

    def __init__(self, debug=False):
        self.debug = debug
        self.unionfind = DisjointSet()
        self.curr_eclass_id = 1 # new eclass id
        # self.nodes = []
        self.node2id = dict()
        self.ids_to_remove = [] # eclass ids to remove
        self.pending = []
        self.analysis_pending = []
        self.id2eclass = dict() # id2eclass

        self.analyses = []
    
    def debug_str(self):
        return ', '.join(str(c) for c in self.id2eclass.values())
    
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
        return Node(enode.op, new_children, enode.metadata)
    
    def get_node_eclass_id(self, enode: Node):
        enode = self.canonicalize(enode)
        return self.node2id[enode]
    
    def get_eclass_by_id(self, eclass_id) -> EClass:
        return self.id2eclass[self.unionfind[eclass_id]]
    
    def get_node_eclass(self, node: Node) -> EClass:
        return self.get_eclass_by_id(self.node2id[node])
    
    def get_node_eclass_id(self, node: Node) -> int:
        return self.node2id[node]
    
    def get_all_eclasses(self):
        return set(self.get_eclass_by_id(v) for v in self.node2id.values())

    def new_singleton_eclass(self, node: Node):
        eclass = EClass(self.curr_eclass_id)
        eclass.nodes.add(node)

        # eclass-analysis
        for a in self.analyses:
            a.make(node, eclass.id)
        
        # bookkeeping
        self.id2eclass[eclass.id] = eclass
        self.curr_eclass_id += 1
        return eclass.id
    
    def add(self, enode: Node):
        """
        Adds a node. Returns the eclass id of the node
        """
        enode = self.canonicalize(enode)
        if enode in self.node2id: # hashcons(?)
            return self.node2id[enode]
        eclass_id = self.new_singleton_eclass(enode)
        for child_eclass_id in enode.children:
            self.id2eclass[child_eclass_id].parents.add(eclass_id)
        self.unionfind.add(eclass_id)
        self.node2id[enode] = eclass_id
        return eclass_id
    
    def merge(self, eid1: int, eid2: int):
        """Merges two eclass ids"""
        logger.debug(f'merging {eid1}, {eid2}')
        if eid1 == eid2:
            return eid1
        # perform_union egraph.rs 1144
        # we can see that at 1177, we actually remove class2 and add class2 nodes to class1
        # WE CAN'T REORDER eid1, eid2 based on parents size because scipy automatically reorders them for their disjoin set anyways...
        # eid1, eid2 = (eid1, eid2) if len(self.id2eclass[eid1].parents) >= len(self.id2eclass[eid2].parents) else (eid2, eid1)
        merged = self.unionfind.merge(eid1, eid2) # I think eid1 will be the new root
        new_id = self.find(eid1)
        if not merged: # xy already merged
            return new_id
    
        analysis_changed = False
        for a in self.analyses:
            analysis_changed = a.join(new_id, [eid1, eid2]) or analysis_changed
        
        if analysis_changed:
            self.analysis_pending.extend(self.id2eclass[eid2].parents)
            self.analysis_pending.extend(self.id2eclass[eid1].parents)
        
        # NOTE in the egg source code, they self.classes.remove id2 and then they do concat_vecs on the nodes and parents
        # SO WHAT TF IS THE POINT OF THE UNION FIND THEN???
        # answer: because the graph is broken, so other stuff still points at the old class. We have to wait till repair
        # after repairing, the eclass for eid2 will be gone.
        
        
        # add class2 to new_id and delete
        # if class1 is not new_id, add class1 to new_id and delete too

        # We extend the worklist with the newly merged in nodes' parents
        # So in the paper, they say they extend with the class and look at parents.
        # here, they extend with parents and look at children I think
        logger.debug(f'{new_id=}, {eid1=} {eid2=}')
        if new_id != eid2:
            self.id2eclass[new_id].nodes = self.id2eclass[new_id].nodes.union(self.id2eclass[eid2].nodes)
            self.id2eclass[new_id].parents = self.id2eclass[new_id].parents.union(self.id2eclass[eid2].parents)
            self.pending.extend(self.id2eclass[eid2].parents)
            self.ids_to_remove.append(eid2)

            # We look at parents in the pending so let's update hashcons for children here
            for n in self.id2eclass[eid2].nodes:
                self.node2id[self.canonicalize(n)] = new_id
        if new_id != eid1:
            self.id2eclass[new_id].nodes = self.id2eclass[new_id].nodes.union(self.id2eclass[eid1].nodes)
            self.id2eclass[new_id].parents = self.id2eclass[new_id].parents.union(self.id2eclass[eid1].parents)
            self.pending.extend(self.id2eclass[eid1].parents)
            self.ids_to_remove.append(eid1)
            for n in self.id2eclass[eid1].nodes:
                self.node2id[self.canonicalize(n)] = new_id
        logger.debug(f'{self.pending=}')
        return new_id
    
    def process_unions(self):
        # egraph.rs 1333
        # TODO maybe find for every parent and remove dupes
        logger.debug(f'processing unions, {self.ids_to_remove=}')
        while len(self.pending):
            eclass_id = self.pending.pop()
            eclass: EClass = self.id2eclass[self.find(eclass_id)]
            nodes = list(eclass.nodes)
            # logger.debug(f'repairing {eclass_id} {nodes}')
            for node in nodes:
                new_node = self.canonicalize(node)
                # logger.debug(f'{node} -> {new_node}')

                already_in_memo = not new_node.matches(node) and new_node in self.node2id
                
                # updates
                if already_in_memo:
                    self.merge(eclass_id, self.node2id[new_node])
                if node in self.node2id:
                    self.node2id.pop(node)
                self.node2id[new_node] = self.find(eclass_id)
                eclass.nodes.remove(node)
                eclass.nodes.add(new_node)
        if self.debug:
            # map should only contain ids of representatives(I added this)
            assert all(k in self.node2id.values() for k in self.id2eclass)

            # hashcons should not point to any outdated e-classes
            assert all(self.find(k) == k for k in self.node2id.values()), self.node2id.values()

            # all items in hashcons should be canonicalized
            for n in self.node2id:
                assert self.canonicalize(n) == n, (self.canonicalize(n), n)
    
    def repair_classes(self):
        # egraph.rs 1249
        logger.debug('processing analysis')
        while len(self.analysis_pending):
            eclass_id = self.find(self.analysis_pending.pop()) # only deal with most updated
            eclass: EClass = self.id2eclass[eclass_id]
            # nodes = list(eclass.nodes)
            data_changed = False
            for a in self.analyses:
                data_changed = a.modify(self, eclass_id, eclass) or data_changed
                data_changed = a.recreate_analysis(eclass_id, eclass) or data_changed
            if data_changed:
                self.analysis_pending.extend(self.id2eclass[eclass_id].parents)
    
    def remove_ids(self):
        for eid in self.ids_to_remove:
            if eid in self.id2eclass:
                self.id2eclass.pop(eid)
            for a in self.analyses:
                a.remove(eid)

    def rebuild(self):
        # this works because process unions will repair the graph completely,
        # and then repair classes might mess up the graph again but at least it operates on a fixed graph
        while len(self.pending) or len(self.analysis_pending):
            self.process_unions()
            self.repair_classes()
        self.remove_ids()
            



# Extraction, I'm going to put the e-graph into a different format
class ExENode:
    op: str
    children: list # initially a list of int, and then we replace them all with e-classes
    def __init__(self, op, children, metadata):
        self.op = op
        self.children = children
        self.metadata = dict(metadata)
    
    def __getitem__(self, key):
        return self.metadata[key]
    
    def __repr__(self):
        assert all(isinstance(c, ExEClass) for c in self.children)
        return f'xNode({self.op}, {tuple(c.id for c in self.children)})' if self.children else f'xNode({self.op})'

class ExtractedEnode:
    """
    After extraction process
    """
    op: str
    children: list
    metadata: Hashabledict
    def __init__(self, op, children=None, metadata=None):
        if metadata is None:
            metadata = dict()
        self.op = op
        self.children = children if children is not None else []
        self.metadata = metadata
    
    @staticmethod
    def from_enode(enode: ExENode):
        return ExtractedEnode(enode.op, enode.children, enode.metadata)
    
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
    
    def __getitem__(self, key):
        return self.metadata[key]
    
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
        x_enode = ExENode(n.op, n.children, n.metadata)
        x_eclass.add_node(x_enode)
        x_enodes.append(x_enode)
    
    for n in x_enodes:
        n.children = tuple(eclasses[egraph.unionfind[c]] for c in n.children)
    return eclasses, x_enodes

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



# Observations
# 1. ok like why are we just holding a bunch of nodes that we're not using?
# when we merge, we merge class1.nodes, class2.nodes. Do we even need a union find then??
# 2. you might not actually have some form of nodes in node2id, so we added
# if node in self.node2id: self.node2id.pop(node) to process_unions. not sure when this would happen...
# 3. extraction process is...interesting, because you might have e.g. x * 0 = 0, x*0 and 0 are in the same class so you are recursing on yourself
# 4. currently for extraction, we move into a different format, so we actually have a tree. Then, we do the scoring with a local cost function, then the extracted output is a different node format. Pretty weird.

# next steps:
# - change the format of the graph, this is kinda dumb
# - add analysis for joins and add that to rebuilding
# - add the rewrite VM

# current case
# - union, you still need to copy your children and stuff, but you have to support 
# a new data struct just to keep track of what's where cuz it's so messy
# - you have to update the hashcons too, but that needs to happen either way...