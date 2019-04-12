def itersubclasses(cls, seen=None):
    if seen is None:
        seen = set()
    if cls not in seen:
        seen.add(cls)
        for direct_sub in cls.__subclasses__():
            if direct_sub in seen: continue

            yield direct_sub
            for sub in itersubclasses(direct_sub, seen):
                yield sub

def find_subclass(cls, subclass_name, match_case=False):
    process = (lambda s:s) if match_case else (lambda s: s.lower())
    subclass_name = process(subclass_name)
    for subcls in itersubclasses(cls):
        if process(subcls.__name__) == subclass_name:
            return subcls
    return None