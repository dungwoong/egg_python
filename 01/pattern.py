from types import MappingProxyType

# I want a pattern to 
# patterns contain nodes. It binds stuff to each node like the node might have inner variables

d = Hashabledict({'a': 1, 'b': 2})
d2 = Hashabledict({'a': 2, 'b': 1})
print(hash(d) == hash(d2))
print(d, d2)
d['3'] = 2