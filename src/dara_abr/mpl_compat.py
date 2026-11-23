"""Make older matplotlib (< 3.5, e.g. the system 3.3 on many clusters) accept pandas objects with pandas >= 2.

Old matplotlib indexes its inputs as x[:, None], which pandas 2 no longer allows on a Series. Importing this module
wraps the Axes plotting methods so that every pandas Series / Index / DataFrame argument is converted to a NumPy array
first. On recent matplotlib it changes nothing that matters (conversion is what matplotlib does anyway).

Usage (before any plotting):  from dara_abr import mpl_compat  # noqa: F401
"""
import functools
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

_METHODS = ("plot", "scatter", "errorbar", "fill_between", "fill_betweenx", "bar", "barh", "step", "hist", "boxplot",
            "axhline", "axvline", "hlines", "vlines", "text", "annotate", "imshow", "set_xticks", "set_yticks",
            "set_xticklabels", "set_yticklabels")


def _np(v):
    if isinstance(v, (pd.Series, pd.Index)):
        return v.to_numpy()
    if isinstance(v, pd.DataFrame):
        return v.to_numpy()
    if isinstance(v, (list, tuple)) and any(isinstance(x, (pd.Series, pd.Index)) for x in v):
        return type(v)(_np(x) for x in v)
    return v


def _wrap(fn):
    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        return fn(self, *[_np(x) for x in args], **{k: _np(v) for k, v in kwargs.items()})
    inner._dara_compat = True
    return inner


for _m in _METHODS:
    _f = getattr(Axes, _m, None)
    if _f is not None and not getattr(_f, "_dara_compat", False):
        setattr(Axes, _m, _wrap(_f))
