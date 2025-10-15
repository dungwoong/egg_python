from typing import List
from egraph.egraph import EGraph, Node
from egraph.pattern import MatcherProgram, ASTNode, MatcherCompiler
from egraph.util import topology_sort
from egraph.logging import rewrite_logger as logger

# A rewrite consists of a matcher program and an applier
# both M and A will have references to the egraphs so no need to worry

class ApplierProgram:
    def __init__(self, egraph: EGraph, old_ast_roots: List[ASTNode], ast_roots: List[ASTNode], children_first_nodes: List[ASTNode]=None):
        self.to_bind = list(zip(old_ast_roots, ast_roots))
        self.children_first_nodes = children_first_nodes
        self.egraph = egraph
    
    def run(self, bindings):
        node2eclassid = dict() # astnode to eclass id of the node we made
        assert isinstance(self.children_first_nodes, list)

        # Add new pattern
        for node in self.children_first_nodes:
            metadata = dict()
            op = bindings[node.op] if node.op in bindings else node.op
            for k in node.metadata:
                assert node.metadata[k] in bindings or not node.metadata[k].startswith('?')
                metadata[k] = bindings[node.metadata[k]] if node.metadata[k] in bindings else node.metadata[k]
            children = tuple(node2eclassid[n] for n in node.children)
            node_to_add = Node(op, children, metadata)
            node2eclassid[node] = self.egraph.add(node_to_add)
        
        for old_root, new_root in self.to_bind:
            # old_root_eclass = self.egraph.get_node_eclass_id(bindings[old_root])
            old_root_eclass = self.egraph.node2id[bindings[old_root]] # don't use canonicalized since graph is broken
            self.egraph.merge(old_root_eclass, node2eclassid[new_root])

class ApplierCompiler:
    """
    If there's anything that's like ?x we look at bindings

    Need to make sure number of roots from matcher vs applier is the same
    we will just merge them one by one
    """
    def __init__(self, egraph: EGraph):
        self.egraph = egraph

    def compile(self, old_ast_roots: List[ASTNode], ast_roots: List[ASTNode]):
        # literally just sort the nodes in the tree and do children first
        toposorted_nodes = topology_sort(ast_roots, lambda x: tuple(x.children), parents_first=False)
        return ApplierProgram(self.egraph, old_ast_roots, ast_roots, toposorted_nodes)


class Rewrite:
    matcher: MatcherProgram
    applier: ApplierProgram
    label: str

    def __init__(self, matcher: MatcherProgram, applier: ApplierProgram, label=""):
        self.matcher = matcher
        self.applier = applier
        self.label = label
    
    def reset(self):
        self.matcher.matches = []
    
    def find_rewrites(self):
        self.matcher.run()
        logger.debug(f'{"no_label" if not self.label else self.label} found {len(self.matcher.matches)} matches')
    
    def apply_rewrites(self):
        for binding in self.matcher.matches:
            self.applier.run(binding)
    
    @staticmethod
    def new(old_ast_roots, new_ast_roots, egraph, label=""):
        matcher_compiler = MatcherCompiler(egraph)
        applier_compiler = ApplierCompiler(egraph)
        matcher = matcher_compiler.compile(old_ast_roots)
        applier = applier_compiler.compile(old_ast_roots, new_ast_roots)
        return Rewrite(matcher, applier, label)