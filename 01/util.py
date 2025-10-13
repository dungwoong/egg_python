class Hashabledict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))
    
    def _blocked(self, *_, **__):
        raise TypeError("HashableDict is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _blocked
