from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import sfHMMBase
from ..single_sfhmm import sfHMM1
from ..multi_sfhmm import sfHMMn

__all__ = ["read_csv", "read_excel", "save_as_csv"]

def check_ref(ref, class_obj):
    if ref is None:
        ref = class_obj()
    elif not isinstance(ref, class_obj):
        raise TypeError(f"`out` must be an object of {class_obj.__name__} or its subclass.")
    return ref

def read_csv(path, out:sfHMMn=None, sep:str=",", encoding:str=None, header="infer",
             **kwargs) -> sfHMMn:
    """
    Read a csv file using pandas.read_csv, and import its data to sfHMMn object.

    Parameters
    ----------
    path : str
        Path to csv file.
    out : sfHMMn or its subclass object, optional
        In which object the content of csv will be stored. If None, a new sfHMMn object
        with default setting will be made. This argument is useful when you prefer your
        own setting of sfHMM input parameter(s).
    sep, encoding, header
        Important arguments in pd.read_csv().
    **kwargs
        Other keyword arguments that will passed to pd.read_csv().

    Returns
    -------
    sfHMMn object
        Object with datasets.
    """    
    out = check_ref(out, sfHMMn)
    df = pd.read_csv(path, sep=sep, encoding=encoding, header=header, **kwargs)
    out.from_pandas(df)
    return out

def read_excel(path:str, ref:sfHMMBase=None, ignore_exceptions:bool=True, sep:str=",", 
               encoding:str=None, header:int=0, sqeeze:bool=False, **kwargs) -> list[sfHMMn]:
    """
    Read a Excel file using pandas.read_excel, and import its data to sfHMMn object. Every sheet
    must have only single trajectory.

    Parameters
    ----------
    path : str
        Path to Excel file.
    ref : sfHMMBase or its subclass object, optional
        From which object the input parameters will be referred. If None, new sfHMMn objects
        with default setting will be made every time. This argument is useful when you prefer 
        your own setting of sfHMM input parameter(s).
    ignore_exceptions : bool
        If True, exceptions will be ignored and only sheets with valid format will be included
        in the output list.
    sep, encoding, header
        Important arguments in pd.read_excel().
    **kwargs
        Other keyword arguments that will passed to pd.read_excel().

    Returns
    -------
    sfHMMn object
        Object with datasets.
    """    
    
    ref = check_ref(ref, sfHMMBase)
    df_dict = pd.read_excel(path, sheet_name=None, sep=sep, encoding=encoding, 
                            header=header, **kwargs)
    
    
    msflist = []
    for sheet_name, df in df_dict.items():
        msf = sfHMMn(**ref.get_params(), name=sheet_name)
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
    elif sqeeze and len(msflist) == 1:
        msflist = msflist[0]
    return msflist
        
# TODO: test
def save_as_csv(obj, path:str):
    df_list = _to_dataframes(obj)
    out = pd.concat(df_list, axis=1)
    out.to_csv(path)
    return None
        
def _to_dataframes(obj:sfHMMBase, suffix:str="") -> list[pd.DataFrame]:
    if isinstance(obj, sfHMM1):
        df = pd.DataFrame(data=obj.data_raw, dtype=np.float64, 
                          columns=["data_raw"],
                          index=np.arange(obj.data_raw.size, dtype=np.int32))
        if hasattr(obj, "step"):
            df[f"step finding-{suffix}"] = obj.step.fit
        if obj.data_fil is not None:
            df[f"denoised-{suffix}"] = obj.data_fil
        if obj.viterbi is not None:
            df[f"Viterbi path-{suffix}"] = obj.viterbi
        return [df]
    
    elif isinstance(obj, sfHMMn):
        df_list = []
        for i, sf in enumerate(obj):
            df_list.append(_to_dataframes(sf, suffix=i)[0])
        return df_list
    else:
        raise TypeError(f"Only sfHMM objects can be converted to pd.DataFrame, but got {type(obj)}")
    
    

    
    