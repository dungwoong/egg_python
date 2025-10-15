class Hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))
    
    def _blocked(self, *_, **__):
        raise TypeError("HashableDict is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _blocked


def topology_sort(items: list, get_children, parents_first=True):
    stack = list(items)
    output_list = []
    started = dict()
    finished = dict()
    time = 0
    while len(stack):
        time += 1
        curr = stack.pop()
        if curr in finished:
            continue
        if curr in started:
            finished[curr] = time
            continue
        output_list.append(curr)
        started[curr] = time
        stack.append(curr)
        children = get_children(curr)
        for child in children:
            stack.append(child)
    output_list = list(sorted(output_list, key=lambda n: finished[n], reverse=parents_first))
    return output_list