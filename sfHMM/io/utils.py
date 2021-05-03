def check_ref(ref, class_obj):
    if ref is None:
        ref = class_obj()
    elif not isinstance(ref, class_obj):
        raise TypeError(f"`out` must be an object of {class_obj.__name__} or its subclass.")
    return ref

def infer_sep(path:str, encoding:str=None):
    if path.endswith(".csv"):
        sep = ","
    elif path.endswith(".tsv"):
        sep = "\t"
    elif path.endswith(".dat"):
        sep = "\s+"
    else:
        with open(path, mode="r", encoding=encoding) as f:
            line0 = f.readline()
            line1 = f.readline()
        if line0.count(",") == line1.count(",") > 0:
            sep = ","
        elif line0.count("\t") == line1.count("\t") > 0:
            sep = "\t"
        else:
            sep = "\s+"
            
    return sep