from util import Hashabledict
from enum import Enum
from egraph import Node, EGraph, EClass
# I want a pattern to 
# patterns contain nodes. It binds stuff to each node like the node might have inner variables

class Instruction(Enum):
    BIND = 'BIND' # bind to a new node. If there's an op, make sure eclass has a node with that op in it
    COMPARE = 'COMPARE' # compare an e-class to another e-class, make sure they have the same id
    LOOKUP = 'LOOKUP' # If we have all children, we can just lookup from children to see if there's a matching pattern to save time
    SCAN = 'SCAN' # literally look through all eclasses to try running the remaining instructions

    def __repr__(self):
        return f'Instruction.{self.value}'

# if we prepend an item with ? then it's a var
# e.g. if metadata is rows: ?x, we bind x. If it's rows: 1, we compare 1
class ASTNode:
    children: list # list of astnodes
    metadata: dict

    def __init__(self, op=None, children=None, metadata=None, label=""):
        self.label = label
        self.op = op # if op is None, that means we can bind
        self.children = [] if children is None else children
        self.metadata = dict() if metadata is None else metadata
    
    def __repr__(self):
        return f'Node({self.label + ", " if self.label else ""}{self.op})'

# so the idea is we just build a tree, and then we can generate an instruction set
def get_variable_bindings(ast_root):
    bindings = dict() # node or var : item
    stack = [ast_root]
    seen = set()
    while stack:
        curr = stack.pop()
        if curr in seen:
            continue
        seen.add(curr)
        bindings[curr] = None
        if curr.op.startswith('?'):
            bindings[curr.op] = None
        for m in curr.metadata.values():
            if m.startswith('?'):
                bindings[m] = None
        stack.extend(curr.children)
    return bindings

def did_yield(gen):
    yielded = False
    for v in gen:
        yielded = True
        yield v
    return yielded

class Program:
    def __init__(self, egraph):
        self.egraph = egraph
        self.instructions = []
        self.bindings = dict()
        self.matches = []
    
    def copy(self):
        # copies everything except for matches(we just add to one main list)
        # and instructions(we just read from same instr set, but we have a ptr)
        p = Program(self.egraph)
        p.instructions = self.instructions
        p.bindings = self.bindings.copy()
        p.matches = self.matches
        return p
    
    def run(self, inst_ptr=None):
        """
        Returns whether we added a match
        """
        # start from the right, and go left
        inst_ptr = len(self.instructions) - 1 if inst_ptr is None else inst_ptr
        if inst_ptr < 0:
            self.matches.append(self.bindings.copy())
            return True
        curr_instr = self.instructions[inst_ptr]
        output = False
        match curr_instr[0]:
            case Instruction.BIND:
                output = output or self.bind(curr_instr[1], inst_ptr)
            case Instruction.SCAN:
                output = output or self.scan(curr_instr[1], inst_ptr)
            case _:
                raise NotImplementedError('Instruction not supported')
        return output

    def execute_instruction(self, instruction: Instruction, args):
        """
        INSTRUCTION SET
        scan and scan_eclass are like for loops
        - SCAN: search all e-classes, iterate through each node
        - SCAN_ECLASS: for an e-class, iterate through all nodes and try to bind or smth
        - BIND: attempt to bind a node to an item
        - [REMOVED]COMPARE: compare a node to another node
        - that lookup thing(later)

        What if we just make bind also process the children. We just set the children
        to whatever they should be binded to, and run a compare if necessary. Otherwise
        we just set the bindings to the e-class for now, and later
        we will call bind again on that child to check the binding.

        So when we're adding this temp thing
        - if there's nothing yet, just add it
        - if the added thing is a node, check that we got the same e-class
        - if the added thing is an e-class, check that we got the same e-class
        """
        match instruction:
            case Instruction.BIND:
                self.bind(*args)
            case _:
                raise NotImplementedError('Instruction not supported')
    
    def bind_node_elements(self, node: Node, reference: ASTNode):
        if reference.op in self.bindings:
            if not self._check_or_bind_item(reference.op, node.op): return False
        for k in reference.metadata:
            if k not in node.metadata: return False
            if not self._check_or_bind_item(reference.metadata[k], node.metadata[k]): return False
        return True

    
    def _check_or_bind_item(self, k, v):
        if self.bindings.get(k, None) is not None and self.bindings.get(k, None) != v:
            return False
        self.bindings[k] = v
        return True
    
    def bind(self, reference: ASTNode, inst_ptr: int):
        """
        self.bindings[reference] should be an EClass

        We loop through the nodes, and for each node we try to bind it
        and continue with our instructions
        """
        assert isinstance(self.bindings[reference], EClass)
        for n in self.bindings[reference].nodes:
            p = self.copy()
            if not p._bind(n, reference):
                return False
            if p.run(inst_ptr - 1): # TODO
                return True # find 1 match, we're good.
        
    def _bind(self, node: Node, reference: ASTNode):
        """
        Binds a node to the reference, binds elements of that node, then
        soft-binds(and compares) children of the node

        Precondition: if multiple things rely on this binding, they all share
        the same e-class. So we can just bind.

        Also, we need to check that each additional variable we're binding
        hasn't already been binded. If they are, we must do compares
        """
        if reference.op not in self.bindings and node.op != reference.op:
            return False
        if reference.op not in self.bindings and len(node.children) != len(reference.children):
            return False
        
        # bind node
        self.bindings[reference] = node # NOTE relies on precondition

        # bind node elements e.g. matrix mnk etc.
        if not self.bind_node_elements(node, reference):
            return False
        
        
        # only if the op wasn't bound, we have to check arity of the op, and children
        # e.g. x * 0 = 0, don't have to check x's children. We just bind to x
        if reference.op in self.bindings:
            return True
        
        # put in soft bindings for the child e-classes
        for node_child_eclassid, ref_child in zip(node.children, reference.children):
            assert ref_child in self.bindings, "Binding children is required if reference.op is not arbitrary"
            node_child_eclass = self.egraph.get_eclass_by_id(node_child_eclassid)
            if self.bindings[ref_child] is None:
                self.bindings[ref_child] = node_child_eclass
            
            # NOTE this is where the compare statement is
            elif isinstance(self.bindings[ref_child], EClass):
                # if e-class isn't the same, we can't get a match
                if self.bindings[ref_child] != node_child_eclass:
                    return False
            elif isinstance(self.bindings[ref_child], Node):
                if self.egraph.get_node_eclass(self.bindings[ref_child]) != node_child_eclass:
                    return False
            else:
                raise ValueError(f"This case isn't handled, {self.bindings[ref_child]}")
        return True

    def scan(self, out_key: ASTNode, inst_ptr: int):
        match_found = False
        for eclass in self.egraph.get_all_eclasses():
            self.bindings[out_key] = eclass
            new_program = self.copy()
            if new_program.run(inst_ptr - 1):
                match_found = True
        return match_found


class Compiler:
    def __init__(self, egraph):
        self.program = Program(egraph)
    
    # in case we have a program class later and want to change the impl
    def get_instructions(self):
        return self.program.instructions
    
    def get_bindings(self):
        return self.program.bindings
    
    def add_new_pattern(self, ast_root):
        bindings = get_variable_bindings(ast_root)
        self.get_bindings().update(bindings)
        # push a scan instruction?
        # in rust, they do for all e-classes look for pattern
        # but isn't that equivalent to just scanning at the start?
        # maybe add a scan if the root is not binded yet
        self.get_instructions().append((Instruction.SCAN, ast_root)) # try to bind eclass to the root


    def compile(self, ast_roots: list): # for multipatterns, you can have multiple ASTs
        stack = list(ast_roots)
        seen = set() # ast nodes that we've seen
        while stack:
            curr = stack.pop()
            assert isinstance(curr, ASTNode)
            if curr in seen:
                continue
            seen.add(curr)
            if curr in ast_roots:
                self.add_new_pattern(curr) # TODO do something here, probably add a scan
            self.get_instructions().append((Instruction.BIND, curr))
            stack.extend(curr.children)
        self.program.instructions = list(reversed(self.program.instructions)) # first instr at end

if __name__ == '__main__':
    a = ASTNode('?A', metadata={'rows': '?m', 'cols': '?k'})
    # exp = ASTNode('exp', children=[a], metadata={'rows': '?m', 'cols': '?k'})
    b = ASTNode('?B', metadata={'rows': '?k', 'cols': '?n'})
    base = ASTNode('mm', children=[a, b], metadata={'rows': '?m', 'cols': '?n'})
    egraph = EGraph()
    a_node = Node('a', metadata={'rows': 2, 'cols': 3})
    b_node = Node('b', metadata={'rows': 3, 'cols': 5})
    egraph.add(a_node)
    egraph.add(b_node)
    mm_node = Node('mm', (egraph.get_node_eclass_id(a_node), egraph.get_node_eclass_id(b_node)), metadata={'rows': 2, 'cols': 5})
    egraph.add(mm_node)
    comp = Compiler(egraph)
    comp.compile([base])
    print(comp.program.instructions)
    # print(comp.program.bindings)
    comp.program.run()
    print(comp.program.matches)
    # print(get_variable_bindings(base))