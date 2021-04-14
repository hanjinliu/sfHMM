import pandas as pd
from ..multi_sfhmm import sfHMMn

__all__ = ["read_csv", "read_excel"]

def check_out(out, class_obj):
    if out is None:
        out = class_obj()
    elif not isinstance(out, class_obj):
        raise TypeError(f"`out` must be an object of {class_obj.__name__} or its subclass.")
    return out

def read_csv(path, out=None, sep=",", encoding=None, header="infer", **kwargs):
    
    out = check_out(out, sfHMMn)
    df = pd.read_csv(path, sep=sep, encoding=encoding, header=header, **kwargs)
    out.from_pandas(df)
    return out

def read_excel(path, out=None, ignore_exceptions=True, sep=",", encoding=None, header=0, **kwargs):
    
    out = check_out(out, sfHMMn)
    df_dict = pd.read_excel(path, sheet_name=None, sep=sep, encoding=encoding, 
                            header=header, **kwargs)
    
    msflist = []
    for sheet_name, df in df_dict.items():
        msf = sfHMMn(sg0=out.sg0, psf=out.psf, krange=out.krange, model=out.model, name=sheet_name)
        try:
            msf.from_pandas(df)
            msflist.append(msf)
        except Exception:
            if ignore_exceptions:
                pass
            else:
                raise
            
    if len(msflist) == 0:
        raise RuntimeError("No sfHMMn object was successfully made.")
    
    return msflist
        
    
    
    