def check_ref(ref, class_obj):
    if ref is None:
        ref = class_obj()
    elif not isinstance(ref, class_obj):
        raise TypeError(f"`out` must be an object of {class_obj.__name__} or its subclass.")
    return ref