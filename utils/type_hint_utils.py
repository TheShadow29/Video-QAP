from typing import Dict, Optional, Union, List, Tuple, Any, Callable
from pathlib import Path
from yacs.config import CfgNode as CN
import numpy as np
from torch import Tensor
from PIL import Image

Pstr = Union[str, Path]
LDct = List[Dict[str, Any]]
Arr = np.ndarray
TArr = Union[Tensor, Arr]
IXer = Union[int, slice, Tensor]
ITArr = Union[Tensor, Arr, Image.Image]


__all__ = [
    "Pstr",
    "LDct",
    "Dict",
    "List",
    "Union",
    "Optional",
    "Callable",
    "Any",
    "Tuple",
    "CN",
    "Arr",
    "TArr",
    "IXer",
    "ITArr",
]
