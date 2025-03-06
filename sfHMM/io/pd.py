from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
from sfHMM.io.utils import check_ref, infer_sep
from sfHMM.base import sfHMMBase
from sfHMM.single_sfhmm import sfHMM1
from sfHMM.multi_sfhmm import sfHMMn
from warnings import warn

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["read", "read_excel", "save"]

EXCEL_TESTED = (".xlsx", ".xlsm", ".xltx", ".xltm", ".xls")
_EXCEL_SUFFIXES = frozenset(
    [
        ".xlsx",
        ".xlsm",
        ".xlsb",
        ".xltx",
        ".xltm",
        ".xls",
        ".xlt",
        ".xml",
        ".xlam",
        ".xla",
        ".xlw",
        ".xlr",
    ]
)


def read(
    path: str,
    out: sfHMMn | None = None,
    sep: str = None,
    encoding: str = None,
    header: int = 0,
    **kwargs,
) -> sfHMMn:
    """
    Read a file using appropriate function, and import its data to sfHMMn object. This
    function can read many types of files.

    Parameters
    ----------
    path : str
        Path to file.
    out : sfHMMn or its subclass object, optional
        In which object the content of the file will be stored. If None, a new sfHMMn object
        with default setting will be made. This argument is useful when you prefer your
        own setting of sfHMM input parameter(s).
    sep, encoding, header
        Important arguments. Default is header=0 rather than "infer" because header="infer"
        usually works in a wrong way.
    **kwargs
        Other keyword arguments that will passed to pd.read_csv() or pd.read_excel().

    Returns
    -------
    sfHMMn object
        Object with datasets.
    """
    import pandas as pd

    out = check_ref(out, sfHMMn)
    if is_excel_file(path):
        df = _safe_read_excel(path, sheet_name=0, header=header, **kwargs)
    else:
        # infer sep if it was not given
        sep = infer_sep(path, encoding) if sep is None else sep
        df = pd.read_csv(path, sep=sep, encoding=encoding, header=header, **kwargs)
    out.from_pandas(df)
    return out


def read_excel(
    path: str,
    ref: sfHMMBase = None,
    ignore_exceptions: bool = True,
    header: int = 0,
    squeeze: bool = False,
    **kwargs,
) -> dict[str, sfHMMn] | sfHMMn:
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
    header : int, default is 0
        Header index.
    squeeze: bool, default is False
        If Excel file only contains one sheet, then return sfHMMn object instead of redundant
        dictionary like {"Sheet1": <sfHMMn>}. Default is set to False for compatibility.
        Although this argument collides with that in pd.read_excel(), it is not a problem
        because squeeze in pd.read_excel must be False in our usage.
    **kwargs
        Other keyword arguments that will passed to pd.read_excel().

    Returns
    -------
    sfHMMn objects
        Dictionary of objects with datasets
    """
    ext = Path(path).suffix
    if ext in (".csv", ".txt", ".dat"):
        raise ValueError("For csv, txt or dat files, use `read()` function instead.")

    ref = check_ref(ref, sfHMMBase)

    df_dict = _safe_read_excel(path, sheet_name=None, header=header, **kwargs)

    msfdict = {}
    for sheet_name, df in df_dict.items():
        msf = sfHMMn(**ref.get_params(), name=sheet_name)
        try:
            msf.from_pandas(df)
            msfdict[sheet_name] = msf
        except Exception:
            if ignore_exceptions:
                pass
            else:
                raise

    if len(msfdict) == 0:
        raise RuntimeError("No sfHMMn object was successfully made.")
    elif squeeze and len(msfdict) == 1:
        msfdict = msfdict[sheet_name]

    if ext not in EXCEL_TESTED:
        warn(
            f"Extension '{ext}' has yet been tested. "
            f"{', '.join(EXCEL_TESTED)} are recommended.",
            UserWarning,
        )

    return msfdict


def save(obj: sfHMMBase, path: str) -> None:
    """
    Save obj.step.fit, obj.data_fil and obj.viterbi as csv.

    Parameters
    ----------
    obj : sfHMMBase
        sfHMM object to save.
    path : str
        Saving path.
    """
    import pandas as pd

    df_list = _to_dataframes(obj)
    out = pd.concat(df_list, axis=1)
    out.to_csv(path)
    return None


def is_excel_file(path: str | Path):
    return Path(path).suffix in _EXCEL_SUFFIXES


def _safe_read_excel(path, **kwargs):
    # xlrd cannot open xlsx in some versions. Here try openpyxl if possible.
    import pandas as pd

    try:
        out = pd.read_excel(path, **kwargs)
    except Exception:
        if kwargs.get("engine", None) == "openpyxl":
            raise
        kwargs["engine"] = "openpyxl"
        out = pd.read_excel(path, **kwargs)
    return out


def _to_dataframes(obj: sfHMMBase, suffix: str = "") -> list[pd.DataFrame]:
    import pandas as pd

    if isinstance(obj, sfHMM1):
        df = pd.DataFrame(
            data=obj.data_raw,
            dtype=np.float64,
            columns=[f"data_raw{suffix}"],
            index=np.arange(obj.data_raw.size, dtype=np.int32),
        )
        if obj.step is not None:
            df[f"step finding{suffix}"] = obj.step.fit
        if obj.data_fil is not None:
            df[f"denoised{suffix}"] = obj.data_fil
        if obj.viterbi is not None:
            df[f"Viterbi path{suffix}"] = obj.viterbi
        return [df]

    elif isinstance(obj, sfHMMn):
        df_list = []
        for i, sf in enumerate(obj):
            df_list.append(_to_dataframes(sf, suffix=f"-{i}")[0])
        return df_list
    else:
        raise TypeError(
            f"Only sfHMM objects can be converted to pd.DataFrame, but got {type(obj)}"
        )
